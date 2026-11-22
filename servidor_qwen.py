"""Servidor local de voz clonada com Qwen3-TTS acelerado (faster-qwen3-tts), com streaming.
Roda no venv G:\\IA\\Qwen3TTS\\.venv. Vozes: vozes\\<Nome>_ref\\<nome>_ref.wav + <nome>_ref.txt (o texto falado).

POST /falar {"nome": "lucia", "texto": "..."}
  -> corpo em pedaços: [4 bytes tamanho][PCM float32 mono] ... ; cabeçalho X-SR
GET /pronto -> 200 quando carregado
"""
import base64
import glob
import json
import os
import struct
import tempfile
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", r"G:\IA\Qwen3TTS\hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

TMP = tempfile.mkdtemp(prefix="dublagem_ref_")

AQUI = os.path.dirname(os.path.abspath(__file__))
PORTA = 3912
MODELO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEMPERATURA = 0.7   # 0,9 (padrão) às vezes saía robótico
PEDACO = 6          # passos por pedaço (~0,5 s de áudio) — o 1º som sai em ~0,35 s
SEM_CONTATO = 60

modelo = None
vozes = {}  # nome -> (wav, texto)
trava = threading.Lock()
ultimo_contato = [time.time()]


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def carregar():
    global modelo
    t = time.time()
    modelo = FasterQwen3TTS.from_pretrained(MODELO)
    for d in glob.glob(os.path.join(AQUI, "vozes", "*_ref")):
        nome = os.path.basename(d)[:-4]
        wav = os.path.join(d, f"{nome.lower()}_ref.wav")
        txt = os.path.join(d, f"{nome.lower()}_ref.txt")
        if os.path.exists(wav) and os.path.exists(txt):
            vozes[nome.lower()] = (wav, open(txt, encoding="utf-8").read().strip())
            log(f"voz: {nome}")
    modelo.warmup(prefill_len=100)
    for nome, (wav, txt) in vozes.items():  # aquece com frase completa (captura os CUDA graphs)
        for _ in range(2):
            for _c in modelo.generate_voice_clone_streaming(text="Oi, tudo bem com você?", language="Portuguese",
                                                            ref_audio=wav, ref_text=txt, chunk_size=PEDACO,
                                                            temperature=TEMPERATURA):
                pass
    # aquece também o caminho "referência por fala" (2 s a 16 kHz): o 1º pedido desse tipo levava 2,6 s
    if vozes:
        wav, txt = next(iter(vozes.values()))
        a, sr = sf.read(wav, dtype="float32")
        a = a[: int(2.0 * sr)]
        a16 = np.interp(np.arange(0, len(a), sr / 16000), np.arange(len(a)), a).astype(np.float32)
        tmp = os.path.join(TMP, "aquece.wav")
        sf.write(tmp, a16, 16000)
        for _ in range(2):
            for _c in modelo.generate_voice_clone_streaming(text="Oi, tudo bem com você?", language="Portuguese",
                                                            ref_audio=tmp, ref_text=" ".join(txt.split()[:8]),
                                                            chunk_size=PEDACO, temperature=TEMPERATURA):
                pass
    log(f"pronto em {time.time() - t:.0f} s")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        ultimo_contato[0] = time.time()
        ok = modelo is not None and bool(vozes) and pronto.is_set()
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(json.dumps({"pronto": ok, "vozes": list(vozes)}).encode())

    def do_POST(self):
        ultimo_contato[0] = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            nome = req["nome"].lower()
            tmp = None
            if pronto.is_set() and req.get("ref_audio"):
                # referência por fala: a própria fala original (float32, 16 kHz) + transcrição
                a = np.frombuffer(base64.b64decode(req["ref_audio"]), dtype=np.float32)
                tmp = os.path.join(TMP, f"ref_{threading.get_ident()}_{int(time.time() * 1000)}.wav")
                sf.write(tmp, a, int(req.get("ref_sr", 16000)))
                wav, txt = tmp, req.get("ref_text", "")
            elif pronto.is_set() and nome in vozes:
                wav, txt = vozes[nome]
            else:
                self.send_response(404)
                self.end_headers()
                return
            texto = req["texto"]
            self.send_response(200)
            self.send_header("X-SR", "24000")
            self.send_header("Connection", "close")
            self.end_headers()
            t, primeiro, total = time.time(), None, 0
            maximo = int(12 * (len(texto) / 10 + 3))  # 12 passos por segundo de fala; teto pelo texto
            with trava:
                for chunk, sr, _ in modelo.generate_voice_clone_streaming(
                        text=texto, language="Portuguese", ref_audio=wav, ref_text=txt, chunk_size=PEDACO,
                        temperature=TEMPERATURA, max_new_tokens=maximo):
                    a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if primeiro is None:
                        primeiro = time.time() - t
                    total += len(a)
                    self.wfile.write(struct.pack("<I", a.nbytes) + a.tobytes())
                    self.wfile.flush()
            log(f"{nome}{' (expressão da cena)' if tmp else ''}: {texto}  "
                f"[1º som {primeiro or 0:.2f}s | total {time.time() - t:.2f}s para {total / 24000:.1f}s]")
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            log("cliente desistiu (fala atrasada)")
        except Exception as e:
            log(f"erro: {e}")


pronto = threading.Event()

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"Qwen3-TTS em http://127.0.0.1:{PORTA} — carregando...")
    carregar()
    pronto.set()
    ultimo_contato[0] = time.time()
    while time.time() - ultimo_contato[0] < SEM_CONTATO:
        time.sleep(2)
    log("a dublagem fechou (sem contato); desligando.")
    os._exit(0)
