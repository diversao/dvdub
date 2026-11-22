"""Referências por frase inteira: cada frase do Whisper cujas janelas são (quase) todas do mesmo grupo.
Usa a voz separada a 24 kHz. Gera, para cada grupo: original (frases) + fala clonada em PT."""
import json
import os

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

YT = r"G:\IA\Chatterbox\yt"
AQUI = os.path.dirname(os.path.abspath(__file__))
v24, SR24 = sf.read(os.path.join(YT, "vozes_24k.wav"), dtype="float32")
falas = json.load(open(os.path.join(YT, "falas.json"), encoding="utf-8"))
jan = np.load(os.path.join(YT, "janelas.npy")) / 16000.0  # início da janela em segundos
rot = np.load(os.path.join(YT, "rotulos.npy"))
emb = np.load(os.path.join(YT, "emb.npy"))
cen = np.load(os.path.join(YT, "centros.npy"))

# frases "puras": >= 80% das janelas dentro dela no mesmo grupo, 1,5 a 9 s
puras = {0: [], 1: [], 2: []}
for f in falas:
    d = f["fim"] - f["ini"]
    if not 1.5 <= d <= 9:
        continue
    dentro = [j for j, t in enumerate(jan) if f["ini"] - 0.2 <= t and t + 1.5 <= f["fim"] + 0.2]
    if len(dentro) < 1:
        continue
    gs = rot[dentro]
    g = int(np.bincount(gs, minlength=3).argmax())
    pureza = float((gs == g).mean())
    tipico = float(np.mean([emb[j] @ cen[g] for j in dentro]))
    if pureza >= 0.8:
        seg = v24[int(f["ini"] * SR24):int(f["fim"] * SR24)]
        # música que sobrou: energia fora da fala (quantil baixo) alta em relação ao pico
        env = np.convolve(np.abs(seg), np.ones(480) / 480, "same")
        ruido = float(np.percentile(env, 10) / (np.percentile(env, 95) + 1e-6))
        puras[g].append((tipico - ruido, f, seg))

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
SR = m.sr
frase = "Calma, Yago. Eu tô aqui, cara. A gente vai resolver isso juntos, tá bom?"
sil = lambda s: np.zeros(int(SR * s), np.float32)
bip = lambda n: np.concatenate([np.concatenate([0.25 * np.sin(2 * np.pi * 880 * np.arange(int(SR * 0.1)) / SR),
                                                np.zeros(int(SR * 0.1))]) for _ in range(n)]).astype(np.float32)
saida = []
for n, g in enumerate((1, 2, 0), 1):
    lst = sorted(puras[g], key=lambda x: -x[0])
    print(f"grupo {g}: {len(lst)} frases puras")
    ref, total = [], 0.0
    for _, f, seg in lst:
        ref += [seg, np.zeros(int(0.3 * SR24), np.float32)]
        total += len(seg) / SR24
        print(f"   {f['ini']:6.1f}s  {f['txt']}")
        if total >= 12:
            break
    if not ref:
        continue
    r = np.concatenate(ref)
    r = r / (np.abs(r).max() + 1e-6) * 0.9
    rp = os.path.join(YT, f"ref2_grupo_{g}.wav")
    sf.write(rp, r, SR24)
    m.prepare_conditionals(rp, exaggeration=0.5)
    w = m.generate(frase, language_id="pt", exaggeration=0.5, cfg_weight=0.5).squeeze().cpu().numpy()
    saida += [bip(n), sil(0.3), r[:SR24 * 7], sil(0.7), w / (np.abs(w).max() + 1e-6) * 0.85, sil(1.2)]
out = os.path.join(AQUI, "teste", "candidatos_jason2.wav")
sf.write(out, np.concatenate(saida), SR)
os.system(f'ffmpeg -y -loglevel error -i "{out}" -b:a 160k "{out[:-4]}.mp3"')
print("ok")
