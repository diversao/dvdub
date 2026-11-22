"""Teste do Qwen3-TTS acelerado (faster-qwen3-tts) com a voz da Lucia.
Roda no venv G:\\IA\\Qwen3TTS\\.venv. Mede o tempo até o 1º som (streaming) e o total."""
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Qwen3TTS\hf")
import numpy as np
import soundfile as sf
import torch
from faster_qwen3_tts import FasterQwen3TTS

AQUI = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.wav")
REF_TXT = ("Bad luck, I guess. They're saying that you locked up more of us than you can handle, "
           "and now you gotta let some of us go. Does that mean I get Tomas back? "
           "But this was a wake-up call for the both of us.")
OUT = os.path.join(AQUI, "teste", "qwen3")
os.makedirs(OUT, exist_ok=True)
frases = ["Ai, mãe... Não!",
          "Então, você queria me ver só pra se gabar?",
          "Ei, que merda aconteceu com o carro que eu te dei?!",
          "Você vai dar a volta por cima, garota."]

for tam in sys.argv[1:] or ["0.6B", "1.7B"]:
    t = time.time()
    m = FasterQwen3TTS.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{tam}-Base")
    print(f"\n== {tam}: carregou em {time.time() - t:.1f} s")
    t = time.time()
    m.warmup(prefill_len=100)
    for _ in range(2):  # aquece com frase completa (recomendação do projeto)
        m.generate_voice_clone(text="Oi, tudo bem com você?", language="Portuguese", ref_audio=REF, ref_text=REF_TXT)
    print(f"   aquecido em {time.time() - t:.1f} s")
    for i, f in enumerate(frases):
        torch.cuda.synchronize()
        t0 = time.time()
        primeiro, partes, sr = None, [], 24000
        for chunk, sr, _ in m.generate_voice_clone_streaming(text=f, language="Portuguese", ref_audio=REF,
                                                             ref_text=REF_TXT, chunk_size=6):
            if primeiro is None:
                primeiro = time.time() - t0
            partes.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        total = time.time() - t0
        a = np.concatenate(partes)
        sf.write(os.path.join(OUT, f"lucia_{tam}_{i}.wav"), a, sr)
        print(f"   1º som {primeiro:.2f}s | total {total:.2f}s para {len(a) / sr:.1f}s de fala | {f}")
    print(f"   VRAM pico {torch.cuda.max_memory_allocated() // 2 ** 20} MB")
    del m
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
