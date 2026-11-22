"""Mede onde o Chatterbox multilíngue gasta tempo e testa atalhos."""
import os
import time

os.environ.setdefault("HF_HOME", r"G:\IA\Chatterbox\hf")
import torch
import torch.nn.functional as F
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.models.s3tokenizer import drop_invalid_tokens

AQUI = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(AQUI, "vozes", "Lucia_ref", "lucia_ref.wav")
OUT = os.path.join(AQUI, "teste", "perfil")
os.makedirs(OUT, exist_ok=True)
m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
m.prepare_conditionals(REF, exaggeration=0.6)
sync = torch.cuda.synchronize


def gerar(texto, cfg=0.4, passos=None, autocast=False, nome=None):
    text = punc_norm(texto)
    tt = m.tokenizer.text_to_tokens(text, language_id="pt").to(m.device)
    tt = torch.cat([tt, tt], dim=0)
    tt = F.pad(tt, (1, 0), value=m.t3.hp.start_text_token)
    tt = F.pad(tt, (0, 1), value=m.t3.hp.stop_text_token)
    sync(); t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        st = m.t3.inference(t3_cond=m.conds.t3, text_tokens=tt, max_new_tokens=400, temperature=0.8,
                            cfg_weight=cfg, repetition_penalty=2.0, min_p=0.05, top_p=1.0)
    sync(); t1 = time.time()
    st = drop_invalid_tokens(st[0]).to(m.device)
    with torch.inference_mode():
        wav, _ = m.s3gen.inference(speech_tokens=st, ref_dict=m.conds.gen, n_cfm_timesteps=passos)
    sync(); t2 = time.time()
    a = wav.squeeze().float().cpu().numpy()
    if nome:
        sf.write(os.path.join(OUT, nome), a, m.sr)
    return t1 - t0, t2 - t1, len(st), len(a) / m.sr


frase = "Então, você queria me ver só pra se gabar?"
gerar("Oi.")
gerar("Oi.", autocast=True)
for rot, kw in [("padrão (10 passos)", {}),
                ("som em 5 passos", {"passos": 5}),
                ("som em 3 passos", {"passos": 3}),
                ("fala em meia precisão", {"autocast": True}),
                ("meia precisão + 5 passos", {"autocast": True, "passos": 5}),
                ("sem CFG + meia + 5", {"autocast": True, "passos": 5, "cfg": 0.0})]:
    r = []
    for k in range(2):
        r.append(gerar(frase, nome=f"{rot.replace(' ', '_').replace('(', '').replace(')', '')}_{k}.wav", **kw))
    tf = sum(x[0] for x in r) / 2
    ts = sum(x[1] for x in r) / 2
    tok = sum(x[2] for x in r) / 2
    dur = sum(x[3] for x in r) / 2
    print(f"{rot:28s} fala(T3) {tf:5.2f}s ({tok / tf:4.0f} tokens/s) | som(S3Gen) {ts:4.2f}s | total {tf + ts:5.2f}s para {dur:.1f}s de fala")
