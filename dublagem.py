"""Dublagem ao vivo: ouve o jogo (só o som do Chiaki), transcreve o inglês,
traduz para português e fala por cima, abaixando o som original enquanto fala.

Uso:  py -3.11 dublagem.py            (espera o Chiaki abrir)
      py -3.11 dublagem.py --wav x.wav  (teste com um arquivo)
Teclas (funcionam com o Chiaki em foco):
      F8  pausa / continua
      F9  liga / desliga o abaixar do som original
"""
import os as _os
_os.environ.setdefault("USE_TF", "0")            # transformers sem TensorFlow (carrega mais rápido)
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
import asyncio
import ctypes
import io
import os
import queue
import re
import subprocess
import sys
import threading
import time

try:  # DLLs CUDA 12 (cuBLAS/cuDNN) instaladas pelo pip, exigidas pelo faster-whisper
    import nvidia
    for _sub in ("cublas", "cudnn", "cuda_nvrtc"):
        _d = os.path.join(nvidia.__path__[0], _sub, "bin")
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

AQUI = os.path.dirname(os.path.abspath(__file__))
CAPTURA = os.path.join(AQUI, "captura", "captura.exe")
LOG = os.path.join(AQUI, "dublagem.log")
SR = 16000

# ---------------- ajustes ----------------
PROCESSO = "chiaki.exe"
MODO = "legenda"             # "legenda" = só o que aparece legendado; "audio" = traduz tudo que ouvir
MODELO = "large-v3-turbo"
MOTOR_VOZ = "kokoro"         # "kokoro" (local, ~0,1 s) ou "microsoft" (internet, ~2 s)
KOKORO_HOMEM, KOKORO_MULHER = "pm_alex", "pf_dora"
# Voz rápida própria por personagem (nome da voz do Kokoro ou mistura "voz1+voz2").
# Lucia: escolhida de ouvido em 22/11 entre as candidatas de teste\candidatas_lucia.mp3.
KOKORO_POR_NOME = {"lucia": "pf_dora+af_heart", "jason": "pm_santa+am_michael"}
KOKORO_VELOCIDADE = 1.15
VOZ_HOMEM = "pt-BR-AntonioNeural"
VOZ_MULHER = "pt-BR-FranciscaNeural"
VELOCIDADE = "+15%"          # fala um pouco mais rápido para acompanhar o jogo
VOLUME_VOZ = 1.0             # 1.0 = normal
VOLUME_JOGO_FALANDO = 0.35   # som original enquanto a voz em português fala
SILENCIO_FIM = 0.35          # segundos de silêncio que encerram uma frase
FRASE_LONGA = 3.5            # a partir daqui, corta na maior pausa interna
MAX_FRASE = 6.0              # corta de qualquer jeito
VELOCIDADE_MAX = 1.6         # a voz acelera até isso para caber no tempo da fala original
FALA_VELHA = 3.5             # segundos: fala que ficou para trás disso é descartada
FILA_MAX = 2                 # falas atrasadas acima disso são descartadas
# -----------------------------------------

LIXO = {
    "", "you", "thank you", "thanks for watching", "thank you for watching", "bye", "uh", "um",
    "oh", "ah", "hmm", "mm", "huh", "okay", "so", "music", "applause", "laughter",
    "mmhmm", "mhm", "mmhm", "uhhuh", "uhuh", "hmhm", "huhuh", "mmm", "uhh", "ahh", "ohh", "yeah",
    "hey", "whoa", "wow", "ow", "ugh",
}

pausado = threading.Event()
abaixar = threading.Event()
abaixar.set()
sair = threading.Event()
voz_pronta = threading.Event()
whisper_pronto = threading.Event()


def log(msg):
    linha = time.strftime("%H:%M:%S ") + msg
    print(linha, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(linha + "\n")
    except OSError:
        pass


# ---------------- volume do jogo (ducking) ----------------
class VolumeJogo:
    def __init__(self, pid):
        self.pid = pid
        self.sessoes = []
        try:
            from pycaw.pycaw import AudioUtilities
            import comtypes
            comtypes.CoInitialize()
            self.sessoes = [s for s in AudioUtilities.GetAllSessions()
                            if s.Process and s.Process.pid == pid]
        except Exception as e:  # sem pycaw ou sem sessão: segue sem abaixar
            log(f"(aviso) não consegui controlar o volume do jogo: {e}")
        # Volume "normal" é sempre 100%. NÃO ler o volume atual: se o programa foi fechado com o
        # jogo abaixado, a leitura viria baixa e cada reinício abaixaria mais (chegou a 1,5%).
        self.original = [1.0] * len(self.sessoes)
        self.restaurar()

    def set(self, frac):
        pid = achar_pid(PROCESSO)
        if pid and pid != self.pid:  # o Chiaki foi reaberto: pega as sessões de áudio novas
            self.__init__(pid)
        for s, v in zip(self.sessoes, self.original):
            try:
                s.SimpleAudioVolume.SetMasterVolume(max(0.0, min(1.0, v * frac)), None)
            except Exception:
                pass

    def restaurar(self):
        self.set(1.0)


# ---------------- entrada de áudio ----------------
def achar_pid(nome):
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {nome}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, creationflags=0x08000000).stdout
    m = re.search(r'"[^"]+","(\d+)"', out)
    return int(m.group(1)) if m else None


def fonte_processo(pid):
    """Gera blocos int16 de 0,1 s com o som do processo."""
    p = subprocess.Popen([CAPTURA, str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         creationflags=0x08000000)
    try:
        while not sair.is_set():
            b = p.stdout.read(3200)
            if not b:
                break
            yield np.frombuffer(b, dtype=np.int16)
    finally:
        p.kill()


def fonte_wav(caminho):
    a, sr = sf.read(caminho, dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        idx = np.arange(0, len(a), sr / SR)
        a = np.interp(idx, np.arange(len(a)), a)
    a = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(a), 1600):
        yield a[i:i + 1600]
        time.sleep(0.1)
    for _ in range(15):  # silêncio no fim para fechar a última frase
        yield np.zeros(1600, np.int16)
        time.sleep(0.1)


# ---------------- corte de frases (VAD) ----------------
def cortar_frases(fonte, saida):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opts = VadOptions(threshold=0.5, min_silence_duration_ms=150, speech_pad_ms=80,
                      min_speech_duration_ms=200)
    buf = np.zeros(0, np.float32)
    ultimo = 0.0
    for bloco in fonte:
        if pausado.is_set():
            buf = np.zeros(0, np.float32)
            continue
        buf = np.concatenate([buf, bloco.astype(np.float32) / 32768])
        agora = time.time()
        if agora - ultimo < 0.15:
            continue
        ultimo = agora
        ts = get_speech_timestamps(buf, opts)
        if not ts:
            buf = buf[-SR:]  # guarda só 1 s de silêncio
            continue
        ini = max(0, ts[0]["start"] - int(0.05 * SR))
        fim_fala = ts[-1]["end"]
        silencio = (len(buf) - fim_fala) / SR
        dur = (fim_fala - ini) / SR
        corte = None
        if silencio >= SILENCIO_FIM:
            corte = min(len(buf), fim_fala + int(0.05 * SR))
        elif dur >= FRASE_LONGA and len(ts) > 1:
            # fala comprida: corta na maior pausa interna (respiração entre frases)
            gaps = [(ts[i + 1]["start"] - ts[i]["end"], i) for i in range(len(ts) - 1)
                    if (ts[i]["end"] - ini) / SR >= 1.2]
            if gaps:
                g, i = max(gaps)
                corte = (ts[i]["end"] + ts[i + 1]["start"]) // 2
        if corte is None and dur >= MAX_FRASE:
            corte = min(len(buf), fim_fala)
        if corte is not None:
            saida.put((buf[ini:corte].copy(), time.time()))
            buf = buf[corte:]


# ---------------- quem fala: voz grave ou aguda ----------------
CORTE_TOM = 172  # Hz: acima disso, voz feminina (cada personagem usa a mediana das suas falas)


def altura_voz(a):
    """Altura média da voz em Hz (algoritmo YIN), ou None se não deu para medir."""
    q, lo, hi = int(0.04 * SR), SR // 350, SR // 65
    lags = np.arange(1, hi + 1)
    f0s = []
    for i in range(0, len(a) - q - hi, q):
        x = a[i:i + q + hi]
        if np.sqrt((x[:q] ** 2).mean()) < 0.01:
            continue
        # diferença quadrática para cada atraso, via janelas deslizantes
        jan = np.lib.stride_tricks.sliding_window_view(x, q)[1:hi + 1]
        d = ((jan - x[:q]) ** 2).sum(axis=1)
        cm = d * lags / np.maximum(np.cumsum(d), 1e-9)
        cand = np.where(cm[lo - 1:hi - 1] < 0.12)[0]  # exigente: música/efeitos não passam
        if len(cand) == 0:
            continue
        k = lo - 1 + cand[0]
        while k + 1 < hi - 1 and cm[k + 1] < cm[k]:
            k += 1
        f0s.append(SR / (k + 1))
    if len(f0s) < 5:
        return None
    f0s = np.array(f0s)
    med = float(np.median(f0s))
    # voz de verdade tem tom constante; música/ruído dá valores espalhados
    if (np.percentile(f0s, 75) - np.percentile(f0s, 25)) / med > 0.35:
        return None
    return med


ultima_voz = [VOZ_HOMEM, 0.0]


def voz_por_tom(a):
    """Escolhe a voz pelo tom; sem tom medido, repete a voz da frase anterior se foi há pouco."""
    f0 = altura_voz(a)
    if f0:
        voz = VOZ_MULHER if f0 > CORTE_TOM else VOZ_HOMEM
    elif time.time() - ultima_voz[1] < 4:
        voz = ultima_voz[0]
    else:
        voz = VOZ_HOMEM
    ultima_voz[:] = [voz, time.time()]
    return voz, f0


# ---------------- transcrição ----------------
def transcrever(entrada, saida):
    from faster_whisper import WhisperModel
    log(f"carregando o Whisper {MODELO} na placa de vídeo...")
    modelo = WhisperModel(MODELO, device="cuda", compute_type="float16")
    aquece = (np.random.default_rng(0).standard_normal(2 * SR) * 0.05).astype(np.float32)
    for _ in range(2):
        list(modelo.transcribe(aquece, language="en", beam_size=1, without_timestamps=True)[0])
    whisper_pronto.set()
    voz_pronta.wait()
    log("pronto — pode jogar. F8 pausa, F9 liga/desliga abaixar o jogo.")
    while not sair.is_set():
        a, t0 = entrada.get()
        tw = time.time()
        segs, _ = modelo.transcribe(a, language="en", beam_size=1, temperature=0.0,
                                    condition_on_previous_text=False, without_timestamps=True,
                                    vad_filter=False)
        partes = [s.text.strip() for s in segs
                  if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0 and s.compression_ratio < 2.4]
        texto = " ".join(partes).strip()
        limpo = re.sub(r"[^a-z' ]", "", texto.lower()).strip()
        if limpo in LIXO or len(limpo) < 2:
            continue
        saida.put((texto, a, t0, time.time() - tw))


# ---------------- tradução ----------------
sessao = requests.Session()
ultima_traducao = [0.0]


def manter_conexao():
    """Mantém a conexão com o tradutor aberta (refazer a conexão custa ~0,5 s)."""
    while not sair.is_set():
        time.sleep(5)
        if time.time() - ultima_traducao[0] > 15:
            traduzir("ok")


trava_traducao = threading.Lock()


def traduzir(texto):
    # Uma de cada vez: em sequência o Google responde em ~0,09 s; em paralelo, ~1 s cada.
    with trava_traducao:
        ultima_traducao[0] = time.time()
        return _traduzir(texto)


def _traduzir(texto):
    try:
        r = sessao.get("https://translate.googleapis.com/translate_a/single",
                       params={"client": "gtx", "sl": "en", "tl": "pt", "dt": "t", "q": texto},
                       timeout=4)
        return "".join(p[0] for p in r.json()[0] if p[0])
    except Exception as e:
        log(f"(aviso) tradução falhou: {e}")
        return None


# ---------------- voz ----------------
class Kokoro:
    """Voz local (Kokoro-82M) na placa de vídeo."""
    def __init__(self):
        from kokoro import KPipeline
        self.lock = threading.Lock()
        self._misturas = {}
        self.p = KPipeline(lang_code="p", repo_id="hexgrad/Kokoro-82M", device="cuda")
        for v in (KOKORO_HOMEM, KOKORO_MULHER):  # aquece as duas vozes
            self.falar("Pronto.", v)

    def _voz(self, nome):
        """Nome de voz do Kokoro ou mistura 'a+b' (média das duas)."""
        if "+" not in nome:
            return nome
        if nome not in self._misturas:
            partes = [self.p.load_voice(n.strip()) for n in nome.split("+")]
            self._misturas[nome] = sum(partes) / len(partes)
        return self._misturas[nome]

    def _gerar(self, texto, voz, vel):
        with self.lock:
            return np.concatenate([r.audio.numpy() for r in self.p(texto, voice=self._voz(voz), speed=vel)])

    def falar(self, texto, voz, alvo=None):
        """Gera a fala; se passar de `alvo` segundos, gera de novo mais rápido para caber.
        Um pouco de "energia" pela pontuação: "!" mais rápido e mais alto, "..." mais lento e baixo."""
        dv, vol = energia_da_fala(texto)
        base = KOKORO_VELOCIDADE + dv
        a = self._gerar(texto, voz, base)
        dur = len(a) / 24000
        if alvo and dur > alvo:
            vel = min(VELOCIDADE_MAX, base * dur / alvo)
            if vel > base + 0.05:
                a = self._gerar(texto, voz, vel)
        return np.clip(a * vol, -0.98, 0.98), 24000


def energia_da_fala(t):
    """(ajuste de velocidade, fator de volume) pela pontuação da legenda."""
    dv, vol = 0.0, 1.0
    if "!" in t:
        dv += 0.07
        vol *= 1.2
    if "?!" in t or "!?" in t or t.count("!") >= 2:
        dv += 0.05
        vol *= 1.1
    if re.search(r"\b[A-ZÀ-Ý]{3,}\b", t):  # palavra em maiúsculas = grito
        vol *= 1.15
    if "..." in t or "…" in t:
        dv -= 0.08
        vol *= 0.9
    return dv, min(vol, 1.5)


kokoro = None
tocando_ate = [0.0]  # quando a fala que está tocando termina (relógio do sistema)


def falar_microsoft(pt, voz):
    import edge_tts

    async def _f():
        mp3 = bytearray()
        async for ch in edge_tts.Communicate(pt, voz, rate=VELOCIDADE).stream():
            if ch["type"] == "audio":
                mp3 += ch["data"]
        return bytes(mp3)

    return sf.read(io.BytesIO(asyncio.run(_f())), dtype="float32")


def preparar_fala(texto, a, t0, tw):
    """Traduz e gera a voz de uma frase (roda em paralelo com as outras)."""
    tt = time.time()
    pt = traduzir(texto)
    if not pt:
        return None
    tt = time.time() - tt
    mulher, f0 = voz_por_tom(a)
    mulher = mulher == VOZ_MULHER
    voz = VOZ_MULHER if mulher else VOZ_HOMEM
    tv = time.time()
    try:
        if kokoro:
            # cabe no tempo da fala original, descontando o que ainda falta tocar da fala anterior
            falta_tocar = max(0.0, tocando_ate[0] - time.time())
            alvo = max(0.7, len(a) / SR * 1.05 - falta_tocar)
            audio, sr = kokoro.falar(pt, KOKORO_MULHER if mulher else KOKORO_HOMEM, alvo)
        else:
            audio, sr = falar_microsoft(pt, voz)
    except Exception as e:
        log(f"(aviso) voz falhou: {e}")
        return None
    tv = time.time() - tv
    quem = "♀" if voz == VOZ_MULHER else "♂"
    tom = f"{f0:.0f}Hz" if f0 else "?Hz"
    log(f"{quem} EN: {texto}\n         PT: {pt}\n         "
        f"[ouvir {tw:.2f}s | traduzir {tt:.2f}s | voz {tv:.2f}s | total {time.time() - t0:.1f}s | {tom}]")
    return audio * VOLUME_VOZ, sr


def sintetizar(entrada, saida):
    global kokoro
    from concurrent.futures import ThreadPoolExecutor
    if MOTOR_VOZ == "kokoro":
        try:
            kokoro = Kokoro()
        except Exception as e:
            log(f"(aviso) Kokoro não carregou ({e}); usando a voz da Microsoft.")
    voz_pronta.set()
    pool = ThreadPoolExecutor(max_workers=3)
    while not sair.is_set():
        texto, a, t0, tw = entrada.get()
        saida.put((pool.submit(preparar_fala, texto, a, t0, tw), t0))
        while saida.qsize() > FILA_MAX + 1:  # atrasou demais: joga fora a mais velha
            try:
                saida.get_nowait()
                log("(fala atrasada descartada)")
            except queue.Empty:
                break


def tocar(entrada, vol):
    while not sair.is_set():
        fut, t0 = entrada.get()
        r = fut.result()
        if r is None or pausado.is_set():
            continue
        audio, sr = r
        if time.time() - t0 > FALA_VELHA:  # chegou tarde demais, a cena já passou
            log("(fala atrasada pulada)")
            continue
        espera = time.time() - t0
        if espera > 2.0:
            log(f"(tocou {espera:.1f}s depois do fim da frase — fila)")
        if vol and abaixar.is_set():
            vol.set(VOLUME_JOGO_FALANDO)
        tocando_ate[0] = time.time() + len(audio) / sr + 0.05 * entrada.qsize()
        sd.play(audio, sr)
        sd.wait()
        tocando_ate[0] = 0.0
        if vol and entrada.empty():
            vol.restaurar()


class Voz:
    """Uma fala no mixer. Pode estar completa ou ainda chegando em pedaços (streaming)."""

    def __init__(self, a=None):
        self.buf = np.zeros(0, np.float32) if a is None else a
        self.pos = 0
        self.fechada = a is not None
        self.lock = threading.Lock()

    def add(self, a):
        with self.lock:
            self.buf = np.concatenate([self.buf[self.pos:], a])
            self.pos = 0

    def fechar(self):
        self.fechada = True

    def ler(self, n):
        with self.lock:
            x = self.buf[self.pos:self.pos + n]
            self.pos += len(x)
            return x

    def viva(self):
        return not self.fechada or self.pos < len(self.buf)

    def restante(self):
        return (len(self.buf) - self.pos) / Mixer.SR


ATRASO_MAX_CLONE = 2.0   # s: com a fila mais atrasada que isso, a fala vai pelo Kokoro
RATE_MAX_CLONE = 1.5     # aceleração máxima da voz com emoção


def esticar(a, rate):
    """Acelera o áudio `rate` vezes mantendo o tom (phase vocoder)."""
    if rate <= 1.02 or len(a) < 2048:
        return a
    import math
    import torch
    import torchaudio
    x = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    n_fft, hop = 1024, 256
    jan = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, window=jan, return_complex=True)
    adv = torch.linspace(0, math.pi * hop, spec.shape[0])[..., None]
    out = torchaudio.functional.phase_vocoder(spec, rate, adv)
    y = torch.istft(out, n_fft, hop, window=jan)
    return y.numpy().astype(np.float32)


class Mixer:
    """Mistura as falas dubladas num fluxo de áudio só."""
    SR = 24000

    def __init__(self):
        self.vozes = []
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=self.SR, channels=1, dtype="float32", blocksize=480,
                                      callback=self._cb)
        self.stream.start()

    def _cb(self, out, frames, tempo, status):
        mix = np.zeros(frames, np.float32)
        with self.lock:
            for v in self.vozes:
                x = v.ler(frames)
                mix[:len(x)] += x
            self.vozes = [v for v in self.vozes if v.viva()]
        out[:, 0] = np.clip(mix, -1.0, 1.0)

    @classmethod
    def ajustar(cls, audio, sr):
        a = np.asarray(audio, np.float32).reshape(-1)
        if sr != cls.SR:
            a = np.interp(np.arange(0, len(a), sr / cls.SR), np.arange(len(a)), a).astype(np.float32)
        return a

    def tocar(self, audio, sr):
        a = self.ajustar(audio, sr)
        with self.lock:
            self.vozes.append(Voz(a))
        return len(a) / self.SR

    def abrir(self):
        """Fala que vai chegar em pedaços."""
        v = Voz()
        with self.lock:
            self.vozes.append(v)
        return v

    def ativo(self):
        with self.lock:
            return bool(self.vozes)


falando_ate = {}  # personagem (ou "_todos") -> quando a fala dublada termina
SOBREPOR = False  # True = personagens diferentes falam juntos (testado 22/11: virou bagunça)


def tocar_mixer(entrada, vol):
    """Cada fala toca assim que fica pronta. Personagens diferentes podem falar juntos;
    o mesmo personagem nunca fala por cima de si mesmo."""
    mix = Mixer()

    def abaixar_jogo():
        abaixado, ultimo = False, 0.0
        while not sair.is_set():
            if mix.ativo():
                ultimo = time.time()
                if vol and abaixar.is_set() and not abaixado:
                    vol.set(VOLUME_JOGO_FALANDO)
                    abaixado = True
            elif abaixado and time.time() - ultimo > 0.3:
                if vol:
                    vol.restaurar()
                abaixado = False
            time.sleep(0.05)

    def esperar_e_tocar(fut, t0, nome):
        if not SOBREPOR:
            nome = "_todos"  # um canal só: uma fala por vez (sobrepor virou bagunça)
        r = fut.result()
        if r is None or pausado.is_set():
            return
        audio, sr = r
        livre = falando_ate.get(nome, 0.0) - time.time()
        if livre > 0:
            time.sleep(livre)
        espera = time.time() - t0
        limite = FALA_VELHA + (REF_MAX if isinstance(audio, FluxoClone) else 0.0)
        if espera > limite:
            log(f"(fala atrasada pulada: {nome})")
            if isinstance(audio, FluxoClone):
                audio.cancelar()
            return
        if isinstance(audio, FluxoClone):  # voz clonada chegando em pedaços: toca enquanto gera
            v, guardado, t_som = None, [], 0.0
            for pedaco in audio:
                p = Mixer.ajustar(esticar(pedaco, audio.rate) * VOLUME_VOZ, audio.sr)
                if v is None:
                    # folga: com o jogo puxando a placa o clone gera ~no ritmo da fala; guardar um pouco
                    # antes de começar evita buracos no meio da frase
                    guardado.append(p)
                    if sum(len(x) for x in guardado) / Mixer.SR >= FOLGA_CLONE:
                        v = mix.abrir()
                        v.add(np.concatenate(guardado))
                        t_som = time.time()
                else:
                    v.add(p)
                if v is not None:
                    falando_ate[nome] = time.time() + v.restante() + 0.3
            if v is None:  # frase curta: terminou antes de encher a folga
                v = mix.abrir()
                if guardado:
                    v.add(np.concatenate(guardado))
                t_som = time.time()
            v.fechar()
            falando_ate[nome] = time.time() + v.restante()
            log(f"   ↳ voz clonada: 1º som {audio.primeiro:.2f}s depois de pedir, "
                f"saiu na caixa {t_som - t0:.1f}s depois da legenda")
            return
        dur = mix.tocar(audio, sr)
        falando_ate[nome] = time.time() + dur

    threading.Thread(target=abaixar_jogo, daemon=True).start()
    trava_ordem = threading.Lock()  # mantém a ordem das legendas quando é uma fala por vez
    while not sair.is_set():
        fut, t0, nome = entrada.get()
        if SOBREPOR:
            threading.Thread(target=esperar_e_tocar, args=(fut, t0, nome.lower()), daemon=True).start()
        else:
            with trava_ordem:
                esperar_e_tocar(fut, t0, nome.lower())


# ---------------- teclas ----------------
def teclas(vol):
    user32 = ctypes.windll.user32
    VK_F8, VK_F9 = 0x77, 0x78
    antes = {VK_F8: False, VK_F9: False}
    while not sair.is_set():
        for vk in antes:
            agora = bool(user32.GetAsyncKeyState(vk) & 0x8000)
            if agora and not antes[vk]:
                if vk == VK_F8:
                    if pausado.is_set():
                        pausado.clear(); log("▶ dublagem ligada"); bipe(880)
                    else:
                        pausado.set(); log("⏸ dublagem pausada"); bipe(440)
                        if vol: vol.restaurar()
                else:
                    if abaixar.is_set():
                        abaixar.clear(); log("som do jogo não abaixa mais"); bipe(520)
                        if vol: vol.restaurar()
                    else:
                        abaixar.set(); log("som do jogo abaixa enquanto a voz fala"); bipe(700)
            antes[vk] = agora
        time.sleep(0.05)


def bipe(hz):
    threading.Thread(target=lambda: ctypes.windll.kernel32.Beep(hz, 120), daemon=True).start()


# ================= MODO LEGENDA =================
# Lê a legenda oficial (pt-BR) na tela do Chiaki e fala na hora. Só fala o que tem legenda
# no formato "Nome: fala"; o áudio do Chiaki serve só para medir o tom (voz de homem/mulher).
# Elenco conhecido (vale mais que a medição do tom, que a música do jogo engana)
_H, _M = VOZ_HOMEM, VOZ_MULHER
VOZ_POR_NOME = {
    "lucia": _M, "jason": _H, "mia": _M, "boobie": _H, "dre'quan": _H, "drequan": _H, "quan": _H, "dre": _H,
    "levi": _H, "roxy": _M, "luxe": _M, "bae-luxe": _M, "ambas": _M, "cal": _H, "raul": _H, "brian": _H,
    "lori": _M, "randy": _H, "manny": _H, "yago": _H, "gary": _H, "andy": _H, "pooch": _H, "poochie": _H,
    "tommy": _H, "pedro": _H, "lorena": _M, "viper": _M,
    # rótulos genéricos
    "garçom": _H, "garçonete": _M, "motorista": _H, "segurança": _H, "policial": _H, "homem": _H, "mulher": _M,
    "garota": _M, "menina": _M, "senhora": _M, "senhor": _H, "cara": _H, "moça": _M, "vendedor": _H, "vendedora": _M,
}


def canonico(nome):
    """Nome lido pelo OCR -> chave do personagem. 'DreIQuan' -> "dre'quan", 'Lucla' -> 'lucia'."""
    import difflib
    chave = nome.lower().strip()
    if chave in VOZ_POR_NOME or chave in KOKORO_POR_NOME:
        return chave
    lido = re.sub(r"[^a-z0-9à-ÿ]", "", chave)
    melhor, nota = chave, 0.0
    for k in list(VOZ_POR_NOME) + list(KOKORO_POR_NOME):
        ref = re.sub(r"[^a-z0-9à-ÿ]", "", k)
        if len(ref) < 4:
            continue
        r = difflib.SequenceMatcher(None, lido, ref).ratio()
        if r > nota:
            melhor, nota = k, r
    return melhor if nota >= 0.8 else chave


def voz_pelo_nome(chave):
    """Voz pelo nome do personagem, se conhecido (aceita 'Dre'Quan Priest', 'Boobie Ike'...)."""
    if chave in VOZ_POR_NOME:
        return VOZ_POR_NOME[chave]
    for parte in re.split(r"[\s'’\-]+", chave):
        if parte in VOZ_POR_NOME:
            return VOZ_POR_NOME[parte]
    return None
LETRAS_POR_SEG = 14.0   # ritmo médio de fala, para estimar quanto tempo a legenda dura
CLONES = {"lucia"}              # personagens com voz clonada. Desde 22/11: Qwen3-TTS 1.7B (servidor_qwen.py, streaming,
                         # 1º som ~0,4 s). O Chatterbox (servidor_clone.py) foi aposentado: 2-4,5 s e às vezes robótico.
FOLGA_CLONE = 0.5        # s de áudio clonado guardado antes de começar a tocar (evita buracos; o Qwen gera ~1,7× mais rápido que fala)
# Expressão da cena: a própria fala original (inglês) vira a referência do clone — timbre e emoção
# daquela fala. Custa esperar a fala original (até REF_MAX s) + Whisper. Ligado 22/11 a pedido dele.
EXPRESSAO = set()        # 22/11 13:30: desligado — com muita gente falando acumulava e cortava falas. Ligar: {"lucia", "jason"}
REF_MAX = 2.0            # s da fala original usados como referência (2 s já bastou nos testes)
REF_MIN = 0.7            # abaixo disso (fala curtíssima), usa a referência fixa / voz padrão
WHISPER_REF = "large-v3-turbo"
PY_CLONE = r"G:\IA\Qwen3TTS\.venv\Scripts\python.exe"
SCRIPT_CLONE = "servidor_custom.py"  # CustomVoice (vozes fixas + emoção); clone = "servidor_qwen.py"
URL_CLONE = "http://127.0.0.1:3912"
_clone_ok = [False, 0.0]


def clone_pronto():
    if _clone_ok[0]:
        return True
    if time.time() - _clone_ok[1] < 5:
        return False
    _clone_ok[1] = time.time()
    try:
        _clone_ok[0] = requests.get(URL_CLONE + "/pronto", timeout=1).status_code == 200
        if _clone_ok[0]:
            log("voz clonada pronta (Lucia).")
    except requests.RequestException:
        pass
    return _clone_ok[0]


class FluxoClone:
    """Pedido ao servidor do Qwen3: os pedaços de áudio chegam enquanto ele gera."""

    def __init__(self, nome, texto, extra=None):
        import struct
        self._struct = struct
        self.t_pedido = time.time()
        pedido = {"nome": nome, "texto": texto}
        pedido.update(extra or {})
        self.r = requests.post(URL_CLONE + "/falar", json=pedido, stream=True, timeout=(2, 15))
        self.r.raise_for_status()
        self.sr = int(self.r.headers.get("X-SR", 24000))
        self.primeiro = 0.0
        self.t_primeiro = 0.0
        self.rate = 1.0  # aceleração aplicada na reprodução (sem mudar o tom), para caber no tempo

    def __iter__(self):
        raw = self.r.raw
        while True:
            cab = raw.read(4)
            if len(cab) < 4:
                break
            n = self._struct.unpack("<I", cab)[0]
            dados = raw.read(n)
            if not self.primeiro:
                self.t_primeiro = time.time()
                self.primeiro = self.t_primeiro - self.t_pedido
            yield np.frombuffer(dados, dtype=np.float32).copy()
        self.r.close()

    def cancelar(self):
        try:
            self.r.close()
        except Exception:
            pass


def instrucao_da_fala(t):
    """Instrução de emoção para o CustomVoice, pela pontuação e pelas palavras da legenda."""
    tl = t.lower()
    if re.search(r"\b[A-ZÀ-Ý]{3,}\b", t) or t.count("!") >= 2 or "?!" in t or "!?" in t:
        return "Fale gritando, com muita raiva."
    if "!" in t:
        return "Fale com raiva, irritado e enérgico."
    if any(p in tl for p in ("desculpa", "sinto muito", "saudade", "chorar", "morreu", "triste")):
        return "Fale triste, com a voz embargada."
    if any(p in tl for p in ("shh", "quieto", "silêncio", "baixinho", "sussur")):
        return "Fale sussurrando, baixinho."
    if "..." in t or "…" in t:
        return "Fale hesitante, pensativo, devagar."
    if "?" in t:
        return "Fale curioso, perguntando, natural."
    return "Fale natural, conversando, com naturalidade."


def falar_clone(nome, texto, emocao):
    return FluxoClone(nome, texto, {"instruct": instrucao_da_fala(texto)}), 24000


def falar_clone_expressao(nome, texto, ref_audio, ref_text):
    """Clone com a própria fala original como referência (timbre + emoção da cena)."""
    import base64
    extra = {"ref_audio": base64.b64encode(np.asarray(ref_audio, np.float32).tobytes()).decode(),
             "ref_sr": SR, "ref_text": ref_text}
    return FluxoClone(nome, texto, extra), 24000


def emocao_da_fala(t):
    """Intensidade da emoção pela pontuação da legenda (0,3 calma ... 0,95 gritando)."""
    e = 0.5
    if "!" in t:
        e += 0.2 + 0.1 * min(t.count("!") - 1, 2)
    if "?!" in t or "!?" in t:
        e += 0.1
    if re.search(r"\b[A-ZÀ-Ý]{3,}\b", t):  # palavra em maiúsculas = grito
        e += 0.1
    if "..." in t or "…" in t:
        e -= 0.1
    return round(max(0.3, min(0.95, e)), 2)


def manter_clone_vivo():
    """Avisa o servidor do clone a cada 15 s que a dublagem está aberta (senão ele se desliga)."""
    if not CLONES:
        return  # sem clones, deixa o servidor se desligar sozinho (libera a placa de vídeo)
    while not sair.is_set():
        try:
            requests.get(URL_CLONE + "/pronto", timeout=2)
        except requests.RequestException:
            pass
        time.sleep(15)


def iniciar_servidor_clone():
    if not CLONES or not os.path.exists(PY_CLONE):
        return None
    try:
        if requests.get(URL_CLONE + "/pronto", timeout=1).ok:
            return None  # já está rodando
    except requests.RequestException:
        pass
    logf = open(os.path.join(AQUI, SCRIPT_CLONE.replace(".py", ".log")), "a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log("iniciando a voz clonada da Lucia (leva ~30 s; até lá ela fala com a voz padrão)...")
    return subprocess.Popen([PY_CLONE, os.path.join(AQUI, SCRIPT_CLONE)], cwd=AQUI, stdout=logf,
                            stderr=subprocess.STDOUT, env=env, creationflags=0x08000000)


GUARDAR_VOZES = {"lucia", "jason"}  # guarda o áudio original dessas falas em vozes\<personagem>\ (para clonar)


def _norm(s):
    return re.sub(r"[^a-z0-9à-ÿ]", "", s.lower())


_PALAVRA = re.compile(r"^[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*[.,!?…:;]*$|^[.…!?]+$|^\d+[.,!?]*$")


def limpeza(fala):
    """Fração de palavras com cara de palavra de verdade (lixo de OCR: 'QuetO.te', 'pr-v-•')."""
    toks = fala.replace("...", "… ").split()
    if not toks:
        return 0.0
    boas = sum(1 for w in toks if _PALAVRA.match(w) and not re.search(r"[a-zà-ÿ][A-Z]", w))
    return boas / len(toks)


def main_legenda(q_audio):
    import collections
    import difflib
    import legenda as L

    anel = collections.deque(maxlen=40)  # ~4 s de áudio do Chiaki, blocos de 0,1 s
    tons_nome = collections.defaultdict(list)
    falados = collections.deque(maxlen=10)
    q_leg = queue.Queue()
    grupo = {"leituras": [], "t": 0.0, "emitido": None, "vazios": 0}
    ultima_legenda = [0.0]   # quando a legenda mais recente apareceu (corta a referência na troca de fala)
    whisper_ref = [None]     # Whisper para transcrever a fala original (referência do clone)

    def carregar_whisper():
        if not EXPRESSAO:
            return
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(WHISPER_REF, device="cuda", compute_type="float16")
            list(m.transcribe(np.zeros(SR, np.float32), language="en", beam_size=1)[0])
            whisper_ref[0] = m
            log("Whisper pronto: expressão da cena ligada para " + ", ".join(sorted(EXPRESSAO)) + ".")
        except Exception as e:
            log(f"(aviso) Whisper não carregou ({e}); expressão da cena desligada.")

    def referencia_da_fala(fala, t0):
        """Espera a fala original e devolve (áudio 16 kHz, transcrição EN) para servir de referência."""
        dur = min(REF_MAX, len(fala) / LETRAS_POR_SEG + 0.3)
        fim = t0 + dur
        while time.time() < fim:
            if ultima_legenda[0] > t0 + 0.3:  # outra legenda apareceu: a fala dela acabou
                fim = min(fim, ultima_legenda[0] - 0.05)
                break
            time.sleep(0.05)
        time.sleep(max(0.0, fim - time.time()))
        blocos = [b for (t, b) in list(anel) if t0 - 0.25 <= t <= fim + 0.05]
        if not blocos:
            return None
        som = np.concatenate(blocos).astype(np.float32) / 32768
        if len(som) / SR < REF_MIN or float(np.sqrt((som ** 2).mean())) < 0.004:
            return None
        som = som / (np.abs(som).max() + 1e-6) * 0.9
        segs, _ = whisper_ref[0].transcribe(som, language="en", beam_size=1, condition_on_previous_text=False,
                                            vad_filter=False)
        en = " ".join(s.text.strip() for s in segs).strip()
        if len(re.sub(r"[^a-z]", "", en.lower())) < 3:
            return None
        return som, en

    def parecido(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    def ouvir():
        while not sair.is_set():
            pid = achar_pid(PROCESSO)
            if not pid:
                time.sleep(2)
                continue
            for b in fonte_processo(pid):
                anel.append((time.time(), b))
            time.sleep(1)

    def novo_grupo(t=0.0):
        grupo.update(leituras=[], t=t, emitido=None, vazios=0)

    def ao_ler(t, txt, img):
        """Junta as leituras da mesma legenda e fala a mais consistente (votação)."""
        if pausado.is_set():
            novo_grupo()
            return
        r = L.extrair_falas(txt) if txt else None
        if not r:
            grupo["vazios"] += 1
            if grupo["vazios"] >= 3:  # legenda sumiu
                novo_grupo()
            return
        grupo["vazios"] = 0
        falas = r  # [(nome, fala), ...] — pode haver 2 legendas na tela ao mesmo tempo
        texto_falas = " ".join(f for _, f in falas)
        chave = _norm("".join(n + f for n, f in falas))
        lt = grupo["leituras"]
        ref = grupo["emitido"] or (lt[0][2] if lt else None)
        if ref is not None and parecido(chave, ref) < 0.55:  # outra legenda
            novo_grupo(t)
            lt = grupo["leituras"]
        if not lt:
            grupo["t"] = t
        lt.append((falas, texto_falas, chave))
        if grupo["emitido"] is None and len(lt) >= 3:
            # medoide (a leitura mais parecida com as outras), pesado pela "limpeza" do texto
            melhor = max(lt, key=lambda x: sum(parecido(x[2], y[2]) for y in lt) * (0.5 + limpeza(x[1])))
            if limpeza(melhor[1]) < 0.85 and len(lt) < 7:
                return  # leitura suja (fundo claro): espera mais algumas
            grupo["emitido"] = melhor[2]
            if any(parecido(melhor[2], f) > 0.8 for f in falados):
                return  # já falada
            falados.append(melhor[2])
            ultima_legenda[0] = grupo["t"]
            for nome, fala in melhor[0]:
                q_leg.put((nome, fala, grupo["t"]))

    def escolher_voz(nome, t0):
        chave = canonico(nome)
        conhecida = voz_pelo_nome(chave)
        if conhecida:
            return conhecida, None  # elenco conhecido: nem mede (a música engana o medidor)
        # espera ~0,8 s de fala para medir o tom (só a fala dela, sem o começo da anterior)
        time.sleep(max(0.0, t0 + 0.8 - time.time()))
        blocos = [b for (t, b) in list(anel) if t0 + 0.1 <= t <= t0 + 0.85]
        f0 = None
        if blocos:
            som = np.concatenate(blocos).astype(np.float32) / 32768
            pico = np.abs(som).max()
            if pico > 0.002:  # normaliza: o som do jogo vem baixo e o medidor exige volume
                f0 = altura_voz(som / pico * 0.8)
        if f0:
            tons_nome[chave] = (tons_nome[chave] + [f0])[-5:]  # só as 5 medições mais recentes
        if tons_nome[chave]:
            return (VOZ_MULHER if np.median(tons_nome[chave]) > CORTE_TOM else VOZ_HOMEM), f0
        return VOZ_HOMEM, f0

    def guardar_trecho(nome, fala, t0):
        """Guarda o áudio original da fala (para clonar a voz do personagem depois)."""
        # só o tempo da fala dela (estimado pelo tamanho da legenda), para não pegar outras vozes
        dur = min(6.0, len(fala) / LETRAS_POR_SEG + 0.4)
        time.sleep(max(0.0, t0 + dur - time.time()))
        blocos = [b for (t, b) in list(anel) if t0 - 0.3 <= t <= t0 + dur]
        if len(blocos) < 8:
            return
        som = np.concatenate(blocos)
        if np.sqrt((som.astype(np.float32) ** 2).mean()) < 150:  # quase mudo: não serve de referência
            return
        pasta = os.path.join(AQUI, "vozes", re.sub(r"[^\w -]", "", nome).strip() or "sem_nome")
        os.makedirs(pasta, exist_ok=True)
        base = os.path.join(pasta, time.strftime("%Y%m%d_%H%M%S"))
        sf.write(base + ".wav", som, SR)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(fala)

    def gerar(nome, fala, t0):
        """Gera a fala (clone se o personagem tiver, senão Kokoro). Roda em paralelo com as outras."""
        chave = canonico(nome)
        expressao = chave in EXPRESSAO and whisper_ref[0] is not None and clone_pronto()
        clonar = chave in CLONES and clone_pronto()
        if clonar or expressao:
            f0, voz = None, VOZ_POR_NOME.get(chave, VOZ_MULHER)
        else:
            voz, f0 = escolher_voz(nome, t0)
        mulher = voz == VOZ_MULHER
        tv = time.time()
        audio = None
        ref_en = ""
        if expressao:
            try:
                ref = referencia_da_fala(fala, t0)
                if ref:
                    audio, sr = falar_clone_expressao(chave, fala, ref[0], ref[1])
                    ref_en = f" | ref {len(ref[0]) / SR:.1f}s EN: {ref[1]}"
                    if chave not in VOZ_POR_NOME:  # quem fala define o símbolo pelo tom da própria fala
                        f0 = altura_voz(ref[0])
                        mulher = bool(f0 and f0 > CORTE_TOM)
            except Exception as e:
                log(f"(aviso) expressão falhou ({e}); usando a referência fixa.")
        if audio is None and clonar:
            canal = chave if SOBREPOR else "_todos"
            falta_tocar = max(0.0, falando_ate.get(canal, 0.0) - time.time())
            if falta_tocar > ATRASO_MAX_CLONE:
                clonar = False  # fila muito atrasada: esta fala vai pelo Kokoro (que acelera) para recuperar
                log(f"(fila {falta_tocar:.1f}s atrasada: {nome} vai na voz rápida)")
        if audio is None and clonar:
            try:
                audio, sr = falar_clone(chave, fala, emocao_da_fala(fala))
                # a voz com emoção fala ~10,5 letras/s; acelera (sem mudar o tom) para caber no tempo da cena
                alvo = max(0.8, len(fala) / LETRAS_POR_SEG - falta_tocar)
                audio.rate = float(min(RATE_MAX_CLONE, max(1.0, (len(fala) / 10.5) / alvo)))
                motor = "clone"
            except Exception as e:
                log(f"(aviso) clone falhou ({e}); usando a voz padrão.")
        if audio is None and expressao and not clonar:
            voz, f0 = escolher_voz(nome, t0)  # sem referência da cena e sem voz fixa: Kokoro
            mulher = voz == VOZ_MULHER
        if audio is None:
            canal = chave if SOBREPOR else "_todos"
            falta_tocar = max(0.0, falando_ate.get(canal, 0.0) - time.time())  # fala anterior ainda tocando
            alvo = max(0.8, len(fala) / LETRAS_POR_SEG - falta_tocar)
            try:
                if kokoro:
                    voz_k = KOKORO_POR_NOME.get(chave) or (KOKORO_MULHER if mulher else KOKORO_HOMEM)
                    audio, sr = kokoro.falar(fala, voz_k, alvo)
                else:
                    audio, sr = falar_microsoft(fala, voz)
                motor = "kokoro"
            except Exception as e:
                log(f"(aviso) voz falhou: {e}")
                return None
        tv = time.time() - tv
        quem = "♀" if mulher else "♂"
        tom = f"{f0:.0f}Hz" if f0 else "?Hz"
        if isinstance(audio, FluxoClone):
            log(f"{quem} {nome}: {fala}   [clone Qwen3{' + expressão da cena' if ref_en else ''}{ref_en}]")
            return audio, sr
        log(f"{quem} {nome}: {fala}   [{motor} {tv:.2f}s | {tom} | começou {time.time() - t0:.1f}s depois da legenda]")
        return audio * VOLUME_VOZ, sr

    def processar():
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=3)
        while not sair.is_set():
            nome, fala, t0 = q_leg.get()
            if canonico(nome) in GUARDAR_VOZES:
                threading.Thread(target=guardar_trecho, args=(nome, fala, t0), daemon=True).start()
            # entra na fila de reprodução já, na ordem da legenda; gera em paralelo
            q_audio.put((pool.submit(gerar, nome, fala, t0), t0, nome))

    threading.Thread(target=ouvir, daemon=True).start()
    threading.Thread(target=processar, daemon=True).start()
    threading.Thread(target=carregar_whisper, daemon=True).start()
    L.VigiaLegenda(intervalo=0.1, ao_ver=ao_ler).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="testar com um arquivo de áudio em vez do Chiaki")
    ap.add_argument("--modo", choices=["legenda", "audio"], default=MODO,
                    help="legenda = lê a legenda da tela (padrão); audio = ouve e traduz tudo")
    args = ap.parse_args()

    if args.modo == "legenda" and not args.wav:
        global kokoro
        q_audio = queue.Queue()
        log("modo LEGENDA: só fala o que aparece legendado (\"Nome: fala\").")
        iniciar_servidor_clone()
        threading.Thread(target=manter_clone_vivo, daemon=True).start()
        log("carregando a voz...")
        try:
            kokoro = Kokoro()
        except Exception as e:
            log(f"(aviso) Kokoro não carregou ({e}); usando a voz da Microsoft.")
        while not (pid := achar_pid(PROCESSO)):
            log("esperando o Chiaki abrir...")
            time.sleep(3)
        vol = VolumeJogo(pid)
        threading.Thread(target=tocar_mixer, args=(q_audio, vol), daemon=True).start()
        threading.Thread(target=teclas, args=(vol,), daemon=True).start()
        main_legenda(q_audio)
        log("pronto — pode jogar. F8 pausa, F9 liga/desliga abaixar o jogo.")
        try:
            while True:
                time.sleep(1)
        finally:
            vol.restaurar()

    q_frases, q_textos, q_audio = queue.Queue(), queue.Queue(), queue.Queue()
    threading.Thread(target=transcrever, args=(q_frases, q_textos), daemon=True).start()
    threading.Thread(target=sintetizar, args=(q_textos, q_audio), daemon=True).start()
    threading.Thread(target=manter_conexao, daemon=True).start()

    if args.wav:
        threading.Thread(target=tocar, args=(q_audio, None), daemon=True).start()
        whisper_pronto.wait(); voz_pronta.wait()
        cortar_frases(fonte_wav(args.wav), q_frases)
        time.sleep(6)
        return

    vol = None
    tocador = None
    while True:
        pid = achar_pid(PROCESSO)
        if not pid:
            log("esperando o Chiaki abrir...")
            while not (pid := achar_pid(PROCESSO)):
                time.sleep(3)
        log(f"Chiaki encontrado (pid {pid}), ouvindo o jogo.")
        vol = VolumeJogo(pid)
        if tocador is None:
            tocador = threading.Thread(target=tocar, args=(q_audio, vol), daemon=True)
            tocador.start()
            threading.Thread(target=teclas, args=(vol,), daemon=True).start()
        try:
            whisper_pronto.wait(); voz_pronta.wait()  # não acumula fala enquanto carrega
            cortar_frases(fonte_processo(pid), q_frases)
        finally:
            vol.restaurar()
        log("o som do Chiaki parou (fechou?). Procurando de novo...")
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        sair.set()
