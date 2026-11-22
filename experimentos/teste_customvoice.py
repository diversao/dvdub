"""Qwen3-TTS CustomVoice: vozes fixas + emoção por instrução, em português, com streaming.
Roda no venv G:\\IA\\Qwen3TTS\\.venv. Gera uma comparação por voz: neutra, com raiva, triste, sussurrando."""
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Qwen3TTS\hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "teste", "customvoice")
os.makedirs(OUT, exist_ok=True)
m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
spk = m.get_supported_speakers() if hasattr(m, "get_supported_speakers") else m.model.get_supported_speakers()
print("vozes:", spk)
m.warmup(prefill_len=100)
SR = 24000
sil = lambda s: np.zeros(int(SR * s), np.float32)
bip = lambda n: np.concatenate([np.concatenate([0.25 * np.sin(2 * np.pi * 880 * np.arange(int(SR * 0.1)) / SR),
                                                np.zeros(int(SR * 0.1))]) for _ in range(n)]).astype(np.float32)
frase = "Ei, que merda aconteceu com o carro que eu te dei?! Você disse que tinha compradores vindo..."
emocoes = [("neutra", None), ("com raiva", "Fale com muita raiva, gritando, irritado."),
           ("triste", "Fale triste, com a voz embargada, quase chorando."),
           ("sussurrando", "Fale sussurrando, com medo, bem baixinho.")]
escolhidas = sys.argv[1:] or spk
out = []
for i, s in enumerate(escolhidas, 1):
    out += [bip(i), sil(0.3)]
    for rot, ins in emocoes:
        try:
            torch.cuda.synchronize(); t = time.time(); p1 = None; parts = []
            for ch, sr, _ in m.generate_custom_voice_streaming(text=frase, speaker=s, language="Portuguese",
                                                                instruct=ins, chunk_size=6, temperature=0.7):
                if p1 is None:
                    p1 = time.time() - t
                parts.append(np.asarray(ch, np.float32).reshape(-1))
            a = np.concatenate(parts)
            sf.write(os.path.join(OUT, f"{i}_{s}_{rot.replace(' ', '_')}.wav"), a, sr)
            out += [a / (np.abs(a).max() + 1e-6) * 0.85, sil(0.5)]
            print(f"{i} {s:10s} {rot:12s} 1º som {p1:.2f}s total {time.time() - t:.2f}s para {len(a) / sr:.1f}s")
        except Exception as e:
            print(f"{i} {s} {rot} ERRO {e}")
    out += [sil(0.8)]
sf.write(os.path.join(AQUI, "teste", "customvoice.wav"), np.concatenate(out), SR)
print("VRAM", torch.cuda.max_memory_allocated() // 2 ** 20, "MB")
