"""Teste: voz da Lucia clonada com o Chatterbox Multilingual, com emoção.
Roda no venv G:\\IA\\Chatterbox\\venv. Usa os trechos gravados em vozes\\Lucia\\."""
import glob
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import soundfile as sf
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
nome = sys.argv[1] if len(sys.argv) > 1 else "Lucia"
saida = os.path.join(AQUI, "teste", "clone")
os.makedirs(saida, exist_ok=True)

sys.path.insert(0, AQUI)
from dublagem import altura_voz  # noqa: E402  (mesmo medidor de tom da dublagem)

# referência: separa a voz (Demucs) de cada trecho, descarta os que não têm o tom do personagem
# e junta até ~12 s, dos mais "vozeados" para os menos
from demucs.apply import apply_model
from demucs.pretrained import get_model

sep = get_model("htdemucs").cuda().eval()
FEMININO = nome.lower() in ("lucia",)
trechos = []
for f in sorted(glob.glob(os.path.join(AQUI, "vozes", nome, "*.wav"))):
    a, sr = sf.read(f, dtype="float32")
    if len(a) < sr * 0.8:
        continue
    # demucs trabalha em 44,1 kHz estéreo
    x = np.interp(np.arange(0, len(a), sr / 44100), np.arange(len(a)), a).astype(np.float32)
    t = torch.from_numpy(np.stack([x, x]))[None].cuda()
    with torch.no_grad():
        fontes = apply_model(sep, t, split=True, overlap=0.1)[0]
    voz = fontes[sep.sources.index("vocals")].mean(0).cpu().numpy()
    voz16 = np.interp(np.arange(0, len(voz), 44100 / 16000), np.arange(len(voz)), voz).astype(np.float32)
    pico = np.abs(voz16).max()
    if pico < 1e-3:
        continue
    voz16 = voz16 / pico * 0.9
    f0 = altura_voz(voz16)
    ok = f0 and ((f0 > 175) if FEMININO else (f0 < 170))
    print(f"  {os.path.basename(f)}  tom {f0 or 0:.0f} Hz  {'usa' if ok else 'descarta'}")
    if ok:
        # tira o silêncio das pontas
        env = np.convolve(np.abs(voz16), np.ones(800) / 800, "same")
        idx = np.where(env > 0.03)[0]
        if len(idx):
            voz16 = voz16[max(0, idx[0] - 800):idx[-1] + 800]
        trechos.append(voz16)
if not trechos:
    sys.exit(f"sem trechos bons em vozes\\{nome}")
ref, total = [], 0.0
for v in trechos:
    ref += [v, np.zeros(int(0.25 * 16000), np.float32)]
    total += len(v) / 16000
    if total >= 12:
        break
ref_path = os.path.join(saida, f"ref_{nome}.wav")
sf.write(ref_path, np.concatenate(ref), 16000)
print(f"referência: {len(ref) // 2} trechos, {total:.1f} s -> {ref_path}")
del sep
torch.cuda.empty_cache()

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

t = time.time()
modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"carregou em {time.time() - t:.1f} s")

frases = [
    ("Ai, mãe... Não!", 0.7),
    ("Você vai dar a volta por cima, garota.", 0.5),
    ("Então, você queria me ver só pra se gabar?", 0.6),
    ("Ei, que merda aconteceu com o carro que eu te dei?!", 0.9),
]
modelo.generate("Oi.", language_id="pt", audio_prompt_path=ref_path)  # aquece
for i, (txt, emo) in enumerate(frases):
    for e in (0.3, emo):
        t = time.time()
        wav = modelo.generate(txt, language_id="pt", audio_prompt_path=ref_path, exaggeration=e, cfg_weight=0.4)
        dt = time.time() - t
        a = wav.squeeze().cpu().numpy()
        dur = len(a) / modelo.sr
        f = os.path.join(saida, f"{nome}_{i}_emo{e:.1f}.wav")
        sf.write(f, a, modelo.sr)
        print(f"[{dt:.2f}s para {dur:.1f}s de fala | emoção {e}] {txt}")
print("vram MB", torch.cuda.max_memory_allocated() // 2 ** 20)
