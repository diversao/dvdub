"""Clone (Qwen3 Base, referência curta) + emoção por instrução. Mede estabilidade: cada frase 3x,
tom (Hz) de cada geração — escorregada = tom fora da faixa do personagem."""
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Qwen3TTS\hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS
from dublagem import altura_voz

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "teste", "clone_emocao")
os.makedirs(OUT, exist_ok=True)
VOZES = {
    "lucia": (os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.wav"),
              open(os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.txt"), encoding="utf-8").read().strip(),
              (150, 340)),
    "jason": (os.path.join(AQUI, "vozes", "Jason_ref", "jason_ref.wav"),
              open(os.path.join(AQUI, "vozes", "Jason_ref", "jason_ref.txt"), encoding="utf-8").read().strip(),
              (70, 175)),
}
frases = {"lucia": "Ei, que merda aconteceu com o carro que eu te dei?! Você disse que tinha compradores vindo...",
          "jason": "Ei, quem é aquele cara? Calma, Lucia. A gente vai resolver isso juntos..."}
emocoes = [("neutra", None), ("com raiva", "Fale com muita raiva, gritando, irritado."),
           ("triste", "Fale triste, com a voz embargada."), ("sussurrando", "Fale sussurrando, com medo, baixinho.")]
m = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
m.warmup(prefill_len=100)
SR = 24000
sil = lambda s: np.zeros(int(SR * s), np.float32)
bip = lambda n: np.concatenate([np.concatenate([0.25 * np.sin(2 * np.pi * 880 * np.arange(int(SR * 0.1)) / SR),
                                                np.zeros(int(SR * 0.1))]) for _ in range(n)]).astype(np.float32)


def tom(a):
    b = np.interp(np.arange(0, len(a), SR / 16000), np.arange(len(a)), a).astype(np.float32)
    return altura_voz(b / (np.abs(b).max() + 1e-6) * 0.8)


for temp in (0.5, 0.7):
    print(f"\n== temperatura {temp}")
    out = []
    for n, (nome, (ref, reftxt, faixa)) in enumerate(VOZES.items(), 1):
        out += [bip(n), sil(0.3)]
        escorregadas, total = 0, 0
        for rot, ins in emocoes:
            for rep in range(3):
                torch.cuda.synchronize(); t = time.time(); p1 = None; parts = []
                for ch, sr, _ in m.generate_voice_clone_streaming(text=frases[nome], language="Portuguese",
                                                                  ref_audio=ref, ref_text=reftxt, instruct=ins,
                                                                  chunk_size=6, temperature=temp, top_k=30):
                    if p1 is None:
                        p1 = time.time() - t
                    parts.append(np.asarray(ch, np.float32).reshape(-1))
                a = np.concatenate(parts)
                f0 = tom(a)
                ok = f0 is not None and faixa[0] <= f0 <= faixa[1]
                total += 1
                escorregadas += (0 if ok else 1)
                sf.write(os.path.join(OUT, f"t{temp}_{nome}_{rot.replace(' ', '_')}_{rep}.wav"), a, sr)
                print(f"  {nome:6s} {rot:12s} #{rep} tom {f0 or 0:4.0f}Hz {'ok' if ok else 'ESCORREGOU'} | 1º som {p1:.2f}s total {time.time() - t:.2f}s para {len(a) / sr:.1f}s")
                if rep == 0:
                    out += [a / (np.abs(a).max() + 1e-6) * 0.85, sil(0.5)]
        print(f"  {nome}: {escorregadas} escorregadas em {total}")
        out += [sil(0.8)]
    sf.write(os.path.join(AQUI, "teste", f"clone_emocao_t{temp}.wav"), np.concatenate(out), SR)
