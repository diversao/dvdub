"""Compara o laço original do T3 com o rápido (tempo e áudio)."""
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soundfile as sf
import torch
import torch.nn.functional as F
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.t3.modules.cond_enc import T3Cond
import t3_rapido

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "teste", "rapido")
os.makedirs(OUT, exist_ok=True)
m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
m.prepare_conditionals(os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.wav"), exaggeration=0.6)
t3_rapido.preparar(m.t3)
sync = torch.cuda.synchronize


def tokens(texto):
    tt = m.tokenizer.text_to_tokens(punc_norm(texto), language_id="pt").to(m.device)
    tt = torch.cat([tt, tt], dim=0)
    tt = F.pad(tt, (1, 0), value=m.t3.hp.start_text_token)
    return F.pad(tt, (0, 1), value=m.t3.hp.stop_text_token)


def som(st, nome):
    st = drop_invalid_tokens(st).to(m.device)
    with torch.inference_mode():
        wav, _ = m.s3gen.inference(speech_tokens=st, ref_dict=m.conds.gen)
    a = wav.squeeze().float().cpu().numpy()
    sf.write(os.path.join(OUT, nome), a, m.sr)
    return len(a) / m.sr


frases = ["Então, você queria me ver só pra se gabar?", "Ei, que merda aconteceu com o carro que eu te dei?!",
          "Ai, mãe... Não!", "Mãe... Você disse que tinha compradores vindo..."]
with torch.inference_mode():
    m.t3.inference(t3_cond=m.conds.t3, text_tokens=tokens("Oi."), max_new_tokens=50, cfg_weight=0.4)
    t3_rapido.gerar(m.t3, m.conds.t3, tokens("Oi."), 3, cfg_weight=0.4)
for i, f in enumerate(frases):
    sync(); t = time.time()
    with torch.inference_mode():
        a = m.t3.inference(t3_cond=m.conds.t3, text_tokens=tokens(f), max_new_tokens=400, cfg_weight=0.4,
                           temperature=0.8, repetition_penalty=2.0, min_p=0.05, top_p=1.0)[0]
    sync(); ta = time.time() - t
    da = som(a, f"original_{i}.wav")
    sync(); t = time.time()
    b = t3_rapido.gerar(m.t3, m.conds.t3, tokens(f), len(f), cfg_weight=0.4)
    sync(); tb = time.time() - t
    db = som(b, f"rapido_{i}.wav")
    print(f"original {ta:.2f}s ({len(a) / ta:3.0f} tok/s, {da:.1f}s de fala) | rápido {tb:.2f}s ({len(b) / tb:3.0f} tok/s, {db:.1f}s de fala) | {f}")
