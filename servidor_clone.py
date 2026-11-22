"""Servidor local de voz clonada (Chatterbox Multilingual) para a dublagem.
Roda no venv G:\\IA\\Chatterbox\\venv. Uma voz por personagem, a partir de vozes\\<Nome>_ref\\*.wav
(ou vozes\\Lucia_ref\\lucia_ref.wav).

POST /falar  {"nome": "Lucia", "texto": "...", "emocao": 0.6}
  -> 200, corpo = PCM float32 mono; cabeçalho X-SR = taxa de amostragem
GET /pronto  -> 200 "ok" quando o modelo carregou
"""
import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import torch
import torch.nn.functional as F
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens

AQUI = os.path.dirname(os.path.abspath(__file__))
PORTA = 3911
PASSOS_SOM = 10  # padrão; 5 era ~0,4 s mais rápido mas deixou partes com som de robô

modelo = None
vozes = {}       # nome -> Conditionals
ultimo_contato = [time.time()]
SEM_CONTATO = 60  # s sem a dublagem chamar -> desliga (libera a placa de vídeo)
trava = threading.Lock()


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def referencia(nome):
    p = os.path.join(AQUI, "vozes", f"{nome}_ref", f"{nome.lower()}_ref.wav")
    if os.path.exists(p):
        return p
    fs = sorted(glob.glob(os.path.join(AQUI, "vozes", f"{nome}_ref", "*.wav")))
    return fs[0] if fs else None


def carregar():
    global modelo
    t = time.time()
    modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    for d in glob.glob(os.path.join(AQUI, "vozes", "*_ref")):
        nome = os.path.basename(d)[:-4]
        ref = referencia(nome)
        if ref:
            modelo.prepare_conditionals(ref, exaggeration=0.5)
            vozes[nome.lower()] = modelo.conds
            log(f"voz pronta: {nome} ({os.path.basename(ref)})")
    falar("lucia" if "lucia" in vozes else next(iter(vozes)), "Oi, tudo bem?", 0.5)  # aquece
    log(f"modelo carregado em {time.time() - t:.0f} s")


def falar(nome, texto, emocao):
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    conds = vozes[nome]
    c = conds.t3
    t3c = T3Cond(speaker_emb=c.speaker_emb, cond_prompt_speech_tokens=c.cond_prompt_speech_tokens,
                 emotion_adv=float(emocao) * torch.ones(1, 1, 1)).to(device=modelo.device)
    tt = modelo.tokenizer.text_to_tokens(punc_norm(texto), language_id="pt").to(modelo.device)
    tt = torch.cat([tt, tt], dim=0)
    tt = F.pad(tt, (1, 0), value=modelo.t3.hp.start_text_token)
    tt = F.pad(tt, (0, 1), value=modelo.t3.hp.stop_text_token)
    with trava, torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16, enabled=False):  # fp16 quase não acelerava
            st = modelo.t3.inference(t3_cond=t3c, text_tokens=tt, max_new_tokens=400, temperature=0.8,
                                     cfg_weight=0.4, repetition_penalty=2.0, min_p=0.05, top_p=1.0)
        st = drop_invalid_tokens(st[0]).to(modelo.device)
        wav, _ = modelo.s3gen.inference(speech_tokens=st, ref_dict=conds.gen, n_cfm_timesteps=PASSOS_SOM)
    return wav.squeeze().float().cpu().numpy(), modelo.sr


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        ultimo_contato[0] = time.time()
        ok = modelo is not None and bool(vozes)
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(json.dumps({"pronto": ok, "vozes": list(vozes)}).encode())

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            nome = req["nome"].lower()
            if modelo is None or nome not in vozes:
                self.send_response(404)
                self.end_headers()
                return
            ultimo_contato[0] = t = time.time()
            a, sr = falar(nome, req["texto"], req.get("emocao", 0.5))
            log(f"{nome}: {req['texto']}  [{time.time() - t:.2f}s para {len(a) / sr:.1f}s, emoção {req.get('emocao')}]")
            corpo = a.astype(np.float32).tobytes()
            self.send_response(200)
            self.send_header("X-SR", str(sr))
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)
        except Exception as e:
            log(f"erro: {e}")
            self.send_response(500)
            self.end_headers()


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORTA), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"servidor de voz clonada em http://127.0.0.1:{PORTA} — carregando...")
    carregar()
    ultimo_contato[0] = time.time()
    try:
        while time.time() - ultimo_contato[0] < SEM_CONTATO:
            time.sleep(2)
        log("a dublagem fechou (sem contato); desligando.")
        os._exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
