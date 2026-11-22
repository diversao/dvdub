"""Lê a legenda do jogo na tela do Chiaki com o OCR do Windows (pt-BR)."""
import asyncio
import ctypes
import re
import threading
import time

import numpy as np

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

# Faixa da legenda, em fração da área de vídeo (ajustável).
# No GTA VI pelo Chiaki a legenda fica em ~0,90 da altura; o botão de ação ("SENTAR") em ~0,84.
FAIXA_Y = (0.80, 0.97)
FAIXA_X = (0.12, 0.88)

# Fala legendada = "Nome: texto". Nome com 1 a 3 palavras capitalizadas (ou sigla curta tipo DJ).
# 1ª palavra com maiúscula ("Lucia", "Bae-Luxe", "Dre'Quan") ou sigla de 2-3 letras ("DJ"); até 2 palavras
# seguintes de qualquer caixa ("Pessoas locais", "Boobie Ike"). Sigla de 1 letra não ("C:" do Explorador).
# 2ª letra minúscula (exclui dicas em caixa alta como "SENTAR"); depois vale maiúscula no meio
# ("DreIQuan" = Dre'Quan lido com o apóstrofo virando letra) e hífen/apóstrofo ("Bae-Luxe").
_NOME1 = r"(?:[A-ZÀ-Ý][a-zà-ÿ'’][A-Za-zÀ-ÿ'’]*(?:[\-'’][A-ZÀ-Ýa-zà-ÿ][a-zà-ÿ'’]*)?|[A-Z]{2,3})"
_NOMEN = r"[A-Za-zÀ-ÿ][a-zà-ÿ'’\-]*"
RE_NOME = re.compile(rf"(?<![A-Za-zÀ-ÿ])({_NOME1}(?: {_NOMEN}){{0,2}})\s*:\s*")
RE_FALA = re.compile(rf"(?<![A-Za-zÀ-ÿ])({_NOME1}(?: {_NOMEN}){{0,2}})\s*:\s*(.{{2,}})")


def extrair_falas(txt):
    """Todas as falas legendadas no texto, na ordem: [('Lucia', '...'), ('Bae-Luxe', '...')].
    Duas legendas na tela ao mesmo tempo viram duas falas, cada uma com o seu personagem."""
    if "♪" in txt:
        return []  # música: não dubla
    ms = list(RE_NOME.finditer(txt))
    # a 2ª legenda só começa depois de fim de frase (". ", "? ", "! "); "Mia: Olha: isso" não é 2 falas
    ms = [m for i, m in enumerate(ms) if i == 0 or re.search(r"[.?!…]\s*$", txt[:m.start()])]
    out = []
    for i, m in enumerate(ms):
        fim = ms[i + 1].start() if i + 1 < len(ms) else len(txt)
        nome = m.group(1).strip()
        fala = re.sub(r"\[[^\]]{1,20}\]\s*", "", txt[m.end():fim]).strip()  # tira "[Espanhol]", "[Rádio]"...
        if sum(c.isalpha() for c in fala) >= 2:
            out.append((nome, fala))
    return out


def extrair_fala(txt):
    """('Motorista', 'Quer falar comigo?') ou None se não for uma fala legendada."""
    m = RE_FALA.search(txt)
    if not m:
        return None
    nome, fala = m.group(1).strip(), m.group(2).strip()
    if sum(c.isalpha() for c in fala) < 2:
        return None
    return nome, fala


def so_texto(bgra):
    """Deixa só o texto da legenda (branco) e o nome (cor clara, ex. verde 150,230,200)
    em preto sobre fundo branco liso — o fundo do jogo atrapalha muito o OCR."""
    f = bgra[:, :, :3].astype(np.int16)
    mx, mn = f.max(2), f.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)
    branco = (mn > 140) & (sat < 0.2)  # 140: falas em outro idioma ("[Espanhol]") vêm em cinza-claro
    cor = (mx > 170) & (sat > 0.18) & (mn > 90)
    g = np.where(branco | cor, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g, np.full_like(g, 255)])


class LeitorOCR:
    def __init__(self, idioma="pt-BR"):
        self.eng = OcrEngine.try_create_from_language(Language(idioma))
        if self.eng is None:
            raise RuntimeError(f"OCR do Windows sem o idioma {idioma}")
        self.loop = asyncio.new_event_loop()

    def ler(self, bgra, limpar_fundo=True):
        """bgra: array HxWx4 uint8. Devolve as linhas de texto encontradas."""
        original = bgra
        if limpar_fundo:
            bgra = so_texto(bgra)
        h, w = bgra.shape[:2]
        dw = DataWriter()
        dw.write_bytes(bgra.tobytes())
        sb = SoftwareBitmap.create_copy_from_buffer(dw.detach_buffer(), BitmapPixelFormat.BGRA8, w, h,
                                                    BitmapAlphaMode.PREMULTIPLIED)
        res = self.loop.run_until_complete(self.eng.recognize_async(sb))
        linhas = []
        for ln in res.lines:
            t = bloco_central(ln, w)
            if not t:
                continue
            rs = [wd.bounding_rect for wd in ln.words]
            y0, y1 = min(r.y for r in rs), max(r.y + r.height for r in rs)
            if tem_nota(ln, original):  # letra de música (♪): não dubla
                t = "♪ " + t
            linhas.append((t, y0, y1, na_caixa(ln, original)))
        linhas.sort(key=lambda l: l[1])
        # Blocos: linhas vizinhas (gap pequeno) ficam juntas. A legenda "Nome: fala" (com as suas
        # linhas de continuação) é um bloco; o objetivo da missão ("Vá até os bastidores.") é outro.
        blocos, atual = [], []
        for l in linhas:
            if atual and l[1] - atual[-1][2] > 0.9 * max(l[2] - l[1], atual[-1][2] - atual[-1][1]):
                blocos.append(atual)
                atual = []
            atual.append(l)
        if atual:
            blocos.append(atual)
        com_nome = [b for b in blocos if any(RE_FALA.search(l[0]) for l in b)]
        if com_nome:
            blocos = com_nome[-1:]  # a legenda é o bloco mais baixo com "Nome:"
        else:
            # Com o "fundo da legenda" ligado, a legenda fica numa caixa escura e o objetivo não
            if any(l[3] for b in blocos for l in b):
                blocos = [[l for l in b if l[3]] for b in blocos]
        return [l[0] for b in blocos for l in b]


def tem_nota(linha, bgra):
    """True se há um glifo pequeno (♪) logo antes da 1ª palavra ou logo depois da última.
    O OCR do Windows não reconhece o ♪ e o apaga; a letra de música fica igual a uma fala."""
    rs = [w.bounding_rect for w in linha.words]
    if not rs:
        return False
    h = max(r.height for r in rs)
    y0 = int(max(0, min(r.y for r in rs) - 0.3 * h))
    y1 = int(min(bgra.shape[0], max(r.y + r.height for r in rs) + 0.3 * h))
    xa = int(max(0, min(r.x for r in rs) - 2.5 * h))
    xb = int(min(bgra.shape[1], max(r.x + r.width for r in rs) + 2.5 * h))
    reg = (so_texto(bgra)[:, :, 0] == 0)[y0:y1, xa:xb]  # pixels de texto na faixa da linha
    if reg.size == 0:
        return False
    cols = reg.any(axis=0)
    blobs, ini = [], None
    for i, c in enumerate(list(cols) + [False]):
        if c and ini is None:
            ini = i
        elif not c and ini is not None:
            blobs.append((ini, i))
            ini = None
    palavras = [(r.x - xa, r.x + r.width - xa) for r in rs]
    for a, b in blobs:
        meio = (a + b) / 2
        if any(px0 - 2 <= meio <= px1 + 2 for px0, px1 in palavras):
            continue  # faz parte de uma palavra reconhecida
        larg = b - a
        sub = reg[:, a:b]
        alt = int(sub.any(axis=1).sum())
        area = int(sub.sum())
        # ♪: estreito (< 0,8 h), alto (> 0,45 h), com miolo (área entre 6% e 50% de h²)
        if 0.12 * h <= larg <= 0.8 * h and alt >= 0.45 * h and 0.06 * h * h <= area <= 0.5 * h * h:
            return True
    return False


def na_caixa(linha, bgra):
    """True se o fundo atrás da linha é a caixa escura e lisa da legenda."""
    rs = [w.bounding_rect for w in linha.words]
    if not rs:
        return False
    x0 = int(max(0, min(r.x for r in rs) - 4))
    y0 = int(max(0, min(r.y for r in rs) - 3))
    x1 = int(min(bgra.shape[1], max(r.x + r.width for r in rs) + 4))
    y1 = int(min(bgra.shape[0], max(r.y + r.height for r in rs) + 3))
    reg = bgra[y0:y1, x0:x1, :3].astype(np.int16)
    if reg.size == 0:
        return False
    mx = reg.max(2)
    fundo = mx[mx < 120]  # tira as letras (claras)
    if fundo.size < 0.3 * mx.size:
        return False
    return float(np.median(fundo)) < 70 and float(fundo.std()) < 30  # caixa é semitransparente


RE_RELOGIO = re.compile(r"\b\d{1,2}\s?[:.]\s?\d{2}\b")


def bloco_central(linha, largura):
    """Só o bloco de palavras do centro da linha. A legenda é centralizada; o celular do jogo
    (nome de quem liga + cronômetro "00:38") fica ao lado e o OCR às vezes junta na mesma linha."""
    ws = sorted(((w.bounding_rect.x, w.bounding_rect.width, w.bounding_rect.height, w.text) for w in linha.words),
                key=lambda p: p[0])
    if not ws:
        return ""
    alt = float(np.median([p[2] for p in ws])) or 10.0
    blocos, atual = [], [ws[0]]
    for p in ws[1:]:
        ant = atual[-1]
        if p[0] - (ant[0] + ant[1]) > 2.2 * alt:  # espaço grande = outro bloco
            blocos.append(atual)
            atual = [p]
        else:
            atual.append(p)
    blocos.append(atual)
    meio = largura / 2

    def centro(b):
        return (b[0][0] + b[-1][0] + b[-1][1]) / 2

    b = min(blocos, key=lambda b: abs(centro(b) - meio))
    if abs(centro(b) - meio) > 0.22 * largura:  # nada centralizado nessa linha
        return ""
    txt = " ".join(p[3] for p in b)
    return RE_RELOGIO.sub("", txt).strip()


_pid_cache = [None, 0.0]


def pid_chiaki(nome_proc="chiaki.exe"):
    import subprocess
    if time.time() - _pid_cache[1] > 3:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {nome_proc}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, creationflags=0x08000000).stdout
        m = re.search(r'"[^"]+","(\d+)"', out)
        _pid_cache[:] = [int(m.group(1)) if m else None, time.time()]
    return _pid_cache[0]


def chiaki_na_frente():
    """True se a janela em primeiro plano é do Chiaki (evita ler outras janelas por cima)."""
    pid = pid_chiaki()
    if not pid:
        return False
    user32 = ctypes.windll.user32
    p = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(p))
    return p.value == pid


def area_janela(nome_proc="chiaki.exe"):
    """Retângulo (x, y, w, h) da área cliente da janela do Chiaki na tela."""
    pid = pid_chiaki(nome_proc)
    if not pid:
        return None
    user32 = ctypes.windll.user32
    achado = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        p = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and user32.IsWindowVisible(hwnd):
            r = (ctypes.c_long * 4)()
            user32.GetClientRect(hwnd, r)
            if r[2] * r[3] > 200 * 200:
                pt = (ctypes.c_long * 2)(0, 0)
                user32.ClientToScreen(hwnd, pt)
                achado.append((pt[0], pt[1], r[2], r[3]))
        return True

    user32.EnumWindows(cb, 0)
    return max(achado, key=lambda a: a[2] * a[3]) if achado else None


def area_video(x, y, w, h):
    """Área do vídeo 16:9 dentro da janela (o Chiaki põe faixas pretas se sobrar)."""
    vw, vh = w, int(w * 9 / 16)
    if vh > h:
        vh, vw = h, int(h * 16 / 9)
    return x + (w - vw) // 2, y + (h - vh) // 2, vw, vh


def limpar(linhas):
    txt = " ".join(l.strip() for l in linhas).strip()
    letras = sum(c.isalpha() for c in txt)
    return txt if letras >= 3 else ""


class VigiaLegenda(threading.Thread):
    """Lê a faixa da legenda ~6x por segundo e guarda (hora, texto)."""

    def __init__(self, intervalo=0.16, ao_ver=None):
        super().__init__(daemon=True)
        self.intervalo = intervalo
        self.hist = []          # [(t, texto)]
        self.lock = threading.Lock()
        self.ao_ver = ao_ver
        self.parar = threading.Event()
        self.area = None
        self.ultimo_area = 0

    def run(self):
        import mss
        ocr = LeitorOCR()
        with mss.mss() as sct:
            while not self.parar.is_set():
                t = time.time()
                if t - self.ultimo_area > 3 or self.area is None:
                    a = area_janela()
                    self.area = area_video(*a) if a else None
                    self.ultimo_area = t
                if not self.area:
                    time.sleep(1)
                    continue
                if not chiaki_na_frente():  # outra janela na frente: não lê a tela
                    if self.ao_ver:
                        self.ao_ver(t, "", None)
                    time.sleep(0.3)
                    continue
                vx, vy, vw, vh = self.area
                box = {"left": vx + int(vw * FAIXA_X[0]), "top": vy + int(vh * FAIXA_Y[0]),
                       "width": int(vw * (FAIXA_X[1] - FAIXA_X[0])), "height": int(vh * (FAIXA_Y[1] - FAIXA_Y[0]))}
                try:
                    img = np.array(sct.grab(box))  # BGRA
                    txt = limpar(ocr.ler(img))
                except Exception:
                    txt = ""
                with self.lock:
                    self.hist.append((t, txt))
                    corte = t - 30
                    while self.hist and self.hist[0][0] < corte:
                        self.hist.pop(0)
                if self.ao_ver:
                    self.ao_ver(t, txt, img)
                time.sleep(max(0.0, self.intervalo - (time.time() - t)))

    def textos_entre(self, t_ini, t_fim):
        """Textos de legenda vistos no intervalo (sem repetir)."""
        with self.lock:
            vistos = []
            for t, txt in self.hist:
                if t_ini <= t <= t_fim and txt and txt not in vistos:
                    vistos.append(txt)
            return vistos
