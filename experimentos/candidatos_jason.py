"""Monta referência para cada grupo de voz candidato e gera uma fala em PT com cada um."""
import os
import sys

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

YT = r"G:\IA\Chatterbox\yt"
AQUI = os.path.dirname(os.path.abspath(__file__))
grupos = [int(g) for g in sys.argv[1:]] or [1, 2]
a, sr = sf.read(os.path.join(YT, "vozes_16k.wav"), dtype="float32")
jan = np.load(os.path.join(YT, "janelas.npy"))
rot = np.load(os.path.join(YT, "rotulos.npy"))
emb = np.load(os.path.join(YT, "emb.npy"))
cen = np.load(os.path.join(YT, "centros.npy"))
JAN, PASSO = int(1.5 * sr), int(0.75 * sr)

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
SR = m.sr
frase = "Calma, Yago. Eu tô aqui, cara. A gente vai resolver isso juntos, tá bom?"
saida = []
bip = lambda n: np.concatenate([np.concatenate([0.25 * np.sin(2 * np.pi * 880 * np.arange(int(SR * 0.1)) / SR),
                                                np.zeros(int(SR * 0.1))]) for _ in range(n)]).astype(np.float32)
for n, g in enumerate(grupos, 1):
    # sequências de janelas seguidas do grupo, as mais típicas primeiro
    seqs, atual = [], []
    for j, (i, r) in enumerate(zip(jan, rot)):
        if r == g and (not atual or i - jan[atual[-1]] <= PASSO):
            atual.append(j)
        else:
            if len(atual) >= 2:
                seqs.append(atual)
            atual = [j] if r == g else []
    if len(atual) >= 2:
        seqs.append(atual)
    seqs.sort(key=lambda s: -np.mean([emb[j] @ cen[g] for j in s]))
    ref, total = [], 0.0
    for s in seqs:
        ini, fim = jan[s[0]], jan[s[-1]] + JAN
        ref += [a[ini:fim], np.zeros(int(0.25 * sr), np.float32)]
        total += (fim - ini) / sr
        if total >= 11:
            break
    if not ref:
        print(f"grupo {g}: sem trechos seguidos")
        continue
    r = np.concatenate(ref)
    r = r / np.abs(r).max() * 0.9
    rp = os.path.join(YT, f"ref_grupo_{g}.wav")
    sf.write(rp, r, sr)
    print(f"grupo {g}: referência {total:.1f} s")
    m.prepare_conditionals(rp, exaggeration=0.5)
    w = m.generate(frase, language_id="pt", exaggeration=0.5, cfg_weight=0.4).squeeze().cpu().numpy()
    orig = np.interp(np.arange(0, len(r), sr / SR), np.arange(len(r)), r).astype(np.float32)[:SR * 6]
    saida += [bip(n), np.zeros(int(SR * 0.3), np.float32), orig, np.zeros(int(SR * 0.7), np.float32),
              w / (np.abs(w).max() + 1e-6) * 0.85, np.zeros(int(SR * 1.2), np.float32)]
out = os.path.join(AQUI, "teste", "candidatos_jason.wav")
sf.write(out, np.concatenate(saida), SR)
os.system(f'ffmpeg -y -loglevel error -i "{out}" -b:a 128k "{out[:-4]}.mp3"')
print("ok:", out[:-4] + ".mp3")
