"""Servidor local de voz com emoção: Qwen3-TTS CustomVoice (vozes fixas + instrução), com streaming.
Roda no venv G:\\IA\\Qwen3TTS\\.venv.

POST /falar {"nome": "lucia", "texto": "...", "instruct": "Fale com raiva."}
  -> corpo em pedaços: [4 bytes tamanho][PCM float32 mono] ... ; cabeçalho X-SR
GET /pronto -> 200 quando carregado
"""
import json
import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", r"G:\IA\Qwen3TTS\hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from faster_qwen3_tts import FasterQwen3TTS

PORTA = 3912
MODELO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TEMPERATURA = 0.7
PEDACO = 6
SEM_CONTATO = 60
# personagem -> voz fixa do CustomVoice (aiden, dylan, eric, ryan, uncle_fu | ono_anna, serena, sohee, vivian)
VOZES = {"lucia": "serena", "jason": "ryan"}

modelo = None
trava = threading.Lock()
ultimo_contato = [time.time()]
pronto = threading.Event()


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def carregar():
    global modelo
    t = time.time()
    modelo = FasterQwen3TTS.from_pretrained(MODELO)
    modelo.warmup(prefill_len=100)
    for v in set(VOZES.values()):
        for ins in (None, "Fale com raiva."):
            for _c in modelo.generate_custom_voice_streaming(text="Oi, tudo bem com você?", speaker=v,
                                                             language="Portuguese", instruct=ins,
                                                             chunk_size=PEDACO, temperature=TEMPERATURA):
                pass
    log(f"pronto em {time.time() - t:.0f} s — vozes: {VOZES}")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        ultimo_contato[0] = time.time()
        ok = pronto.is_set()
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(json.dumps({"pronto": ok, "vozes": list(VOZES)}).encode())

    def do_POST(self):
        ultimo_contato[0] = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            nome = req["nome"].lower()
            if not pronto.is_set() or nome not in VOZES:
                self.send_response(404)
                self.end_headers()
                return
            texto, ins = req["texto"], req.get("instruct") or None
            self.send_response(200)
            self.send_header("X-SR", "24000")
            self.send_header("Connection", "close")
            self.end_headers()
            t, primeiro, total = time.time(), None, 0
            maximo = int(12 * (len(texto) / 10 + 3))
            with trava:
                for chunk, sr, _ in modelo.generate_custom_voice_streaming(
                        text=texto, speaker=VOZES[nome], language="Portuguese", instruct=ins,
                        chunk_size=PEDACO, temperature=TEMPERATURA, max_new_tokens=maximo):
                    a = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if primeiro is None:
                        primeiro = time.time() - t
                    total += len(a)
                    self.wfile.write(struct.pack("<I", a.nbytes) + a.tobytes())
                    self.wfile.flush()
            log(f"{nome} [{ins or 'neutra'}]: {texto}  [1º som {primeiro or 0:.2f}s | total {time.time() - t:.2f}s para {total / 24000:.1f}s]")
        except (BrokenPipeError, ConnectionResetError):
            log("cliente desistiu (fala atrasada)")
        except Exception as e:
            log(f"erro: {e}")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"CustomVoice em http://127.0.0.1:{PORTA} — carregando...")
    carregar()
    pronto.set()
    ultimo_contato[0] = time.time()
    while time.time() - ultimo_contato[0] < SEM_CONTATO:
        time.sleep(2)
    log("a dublagem fechou (sem contato); desligando.")
    os._exit(0)
