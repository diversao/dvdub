"""Separa os falantes do vídeo (vozes já sem música) e monta a referência do Jason.
Roda no venv do Chatterbox: usa o codificador de voz (VoiceEncoder) do próprio Chatterbox."""
import json
import os
import re

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import KMeans

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

YT = r"G:\IA\Chatterbox\yt"
AQUI = os.path.dirname(os.path.abspath(__file__))
a, sr = sf.read(os.path.join(YT, "vozes_16k.wav"), dtype="float32")
falas = json.load(open(os.path.join(YT, "falas.json"), encoding="utf-8"))

# janelas de 1,5 s (passo 0,75 s) só onde há voz
JAN, PASSO = int(1.5 * sr), int(0.75 * sr)
env = np.convolve(np.abs(a), np.ones(800) / 800, "same")
limiar = max(0.01, np.percentile(env, 60))
janelas = []
for i in range(0, len(a) - JAN, PASSO):
    seg = env[i:i + JAN]
    if (seg > limiar).mean() > 0.6:
        janelas.append(i)
print(f"{len(janelas)} janelas com voz")

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
ve = m.ve
emb = []
for k in range(0, len(janelas), 64):
    lote = [a[i:i + JAN] for i in janelas[k:k + 64]]
    e = ve.embeds_from_wavs(lote, sample_rate=sr, as_spk=False)
    emb.append(np.asarray(e))
emb = np.concatenate(emb)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

for K in (2, 3):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(emb)
    print(f"K={K}: tamanhos {np.bincount(km.labels_).tolist()}")
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(emb)
rot = km.labels_


def texto_em(t):
    for f in falas:
        if f["ini"] <= t <= f["fim"]:
            return f["txt"]
    return ""


# quem fala "Yago" é o Jason; quem fala "Jason" é o Yago
votos = {c: {"diz_yago": 0, "diz_jason": 0, "n": 0} for c in range(3)}
for i, c in zip(janelas, rot):
    tx = texto_em((i + JAN / 2) / sr).lower()
    votos[c]["n"] += 1
    if re.search(r"\byago\b", tx):
        votos[c]["diz_yago"] += 1
    if re.search(r"\bjason\b", tx):
        votos[c]["diz_jason"] += 1
for c, v in votos.items():
    print(f"grupo {c}: {v}")

# amostra de cada grupo para conferir de ouvido
for c in range(3):
    idx = [i for i, r in zip(janelas, rot) if r == c]
    # as janelas mais "centrais" do grupo (mais parecidas com o centro)
    d = [(float(emb[j] @ km.cluster_centers_[c]), i) for j, (i, r) in enumerate(zip(janelas, rot)) if r == c]
    d.sort(reverse=True)
    trechos = [a[i:i + JAN] for _, i in d[:8]]
    if trechos:
        sf.write(os.path.join(YT, f"grupo_{c}.wav"), np.concatenate(
            [np.concatenate([t, np.zeros(int(0.3 * sr), np.float32)]) for t in trechos]), sr)
np.save(os.path.join(YT, "janelas.npy"), np.array(janelas))
np.save(os.path.join(YT, "rotulos.npy"), rot)
np.save(os.path.join(YT, "emb.npy"), emb)
np.save(os.path.join(YT, "centros.npy"), km.cluster_centers_)
print("amostras: grupo_0.wav, grupo_1.wav, grupo_2.wav em", YT)
