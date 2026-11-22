"""Teste do clone da Lucia com a referência baixada (vozes\\Lucia_ref\\lucia_ref.wav)."""
import os
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

AQUI = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.wav")
OUT = os.path.join(AQUI, "teste", "clone2")
os.makedirs(OUT, exist_ok=True)

t = time.time()
m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"carregou em {time.time() - t:.1f} s")
m.prepare_conditionals(REF, exaggeration=0.5)   # prepara a voz uma vez só (mais rápido por frase)
m.generate("Oi.", language_id="pt")             # aquece

frases = [
    ("Ai, mãe... Não!", 0.7),
    ("Você vai dar a volta por cima, garota.", 0.5),
    ("Então, você queria me ver só pra se gabar?", 0.6),
    ("Ei, que merda aconteceu com o carro que eu te dei?!", 0.9),
]
for i, (txt, emo) in enumerate(frases):
    for e in (0.4, emo):
        t = time.time()
        wav = m.generate(txt, language_id="pt", exaggeration=e, cfg_weight=0.4)
        dt = time.time() - t
        a = wav.squeeze().cpu().numpy()
        sf.write(os.path.join(OUT, f"lucia_{i}_emo{e:.1f}.wav"), a, m.sr)
        print(f"[{dt:.2f}s para {len(a) / m.sr:.1f}s de fala | emoção {e}] {txt}")
print("vram MB", torch.cuda.max_memory_allocated() // 2 ** 20)
