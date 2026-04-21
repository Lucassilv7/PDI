import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _pil_resize(array: np.ndarray, new_w: int, new_h: int, method=Image.NEAREST) -> np.ndarray:
    """Auxiliar: redimensiona via PIL sem cv2."""
    pil = Image.fromarray(array.astype(np.uint8))
    return np.array(pil.resize((new_w, new_h), method), dtype=np.uint8)


# =====================================================================
# TRANSFORMAÇÕES GEOMÉTRICAS (NumPy puro)
# =====================================================================

def translacao(img: np.ndarray, tx: int = 0, ty: int = 0) -> np.ndarray:
    """Translação por deslocamento (tx, ty) em pixels."""
    h, w = img.shape[:2]
    result = np.zeros_like(img)

    # Calcula regiões de origem e destino
    src_x0 = max(0, -tx);  src_x1 = min(w, w - tx)
    dst_x0 = max(0,  tx);  dst_x1 = min(w, w + tx)
    src_y0 = max(0, -ty);  src_y1 = min(h, h - ty)
    dst_y0 = max(0,  ty);  dst_y1 = min(h, h + ty)

    if src_x1 > src_x0 and src_y1 > src_y0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

    return result


def rotacao(img: np.ndarray, angulo: float) -> np.ndarray:
    """
    Rotação em torno do centro da imagem (sem cv2).
    Usa mapeamento inverso pixel a pixel com NumPy.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    rad = np.deg2rad(-angulo)          # sinal negativo = sentido horário padrão
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Grade de coordenadas de destino
    ys, xs = np.mgrid[0:h, 0:w]
    xs_c = xs - cx
    ys_c = ys - cy

    # Mapeamento inverso (destino → origem)
    src_x = (cos_a * xs_c - sin_a * ys_c + cx).astype(np.float32)
    src_y = (sin_a * xs_c + cos_a * ys_c + cy).astype(np.float32)

    src_xi = np.round(src_x).astype(int)
    src_yi = np.round(src_y).astype(int)

    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)

    result = np.zeros_like(img)
    result[ys[valid], xs[valid]] = img[src_yi[valid], src_xi[valid]]
    return result


def escala(img: np.ndarray, fx: float = 1.0, fy: float = 1.0) -> np.ndarray:
    """Escala por fatores fx (largura) e fy (altura) via interpolação bilinear NumPy."""
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * fx)))
    new_h = max(1, int(round(h * fy)))

    # Coordenadas no espaço de origem
    x_src = (np.arange(new_w) / fx).astype(np.float32)
    y_src = (np.arange(new_h) / fy).astype(np.float32)

    x0 = np.floor(x_src).astype(int).clip(0, w - 2)
    y0 = np.floor(y_src).astype(int).clip(0, h - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = (x_src - x0)[:, np.newaxis]   # (new_w, 1)
    dy = (y_src - y0)[:, np.newaxis]   # (new_h, 1)

    # Bilinear para cada canal
    channels = []
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        top    = ch[y0][:, x0] * (1 - dx).T + ch[y0][:, x1] * dx.T   # (new_h, new_w)
        bottom = ch[y1][:, x0] * (1 - dx).T + ch[y1][:, x1] * dx.T
        interp = top * (1 - dy) + bottom * dy
        channels.append(interp)

    return np.clip(np.dstack(channels), 0, 255).astype(np.uint8)


def reflexao(img: np.ndarray, eixo: str = "horizontal") -> np.ndarray:
    """
    Reflexão da imagem.
    eixo: 'horizontal' (espelha esquerda-direita)
          'vertical'   (espelha cima-baixo)
          'ambos'
    """
    if eixo == "horizontal":
        return img[:, ::-1, :].copy()
    elif eixo == "vertical":
        return img[::-1, :, :].copy()
    elif eixo == "ambos":
        return img[::-1, ::-1, :].copy()
    else:
        raise ValueError("eixo deve ser 'horizontal', 'vertical' ou 'ambos'.")


def cisalhamento(img: np.ndarray, shx: float = 0.0, shy: float = 0.0) -> np.ndarray:
    """
    Cisalhamento (shear) por mapeamento inverso.
    shx: cisalhamento horizontal, shy: cisalhamento vertical.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]

    src_x = (xs - shx * ys).astype(np.float32)
    src_y = (ys - shy * xs).astype(np.float32)

    src_xi = np.round(src_x).astype(int)
    src_yi = np.round(src_y).astype(int)

    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)

    result = np.zeros_like(img)
    result[ys[valid], xs[valid]] = img[src_yi[valid], src_xi[valid]]
    return result


# =====================================================================
# ZOOM IN – Replicação e Interpolação Bilinear
# =====================================================================

def zoom_in_replicacao(img: np.ndarray, fator: float) -> np.ndarray:
    """
    Zoom IN por replicação de pixels (nearest-neighbor).
    Cada pixel original é replicado fator×fator vezes.
    """
    h, w = img.shape[:2]
    new_h = int(h * fator)
    new_w = int(w * fator)

    # Índices de origem mapeados por vizinho mais próximo
    src_y = (np.arange(new_h) / fator).astype(int).clip(0, h - 1)
    src_x = (np.arange(new_w) / fator).astype(int).clip(0, w - 1)

    return img[np.ix_(src_y, src_x)]


def zoom_in_interpolacao(img: np.ndarray, fator: float) -> np.ndarray:
    """
    Zoom IN por interpolação bilinear (sem cv2).
    """
    return escala(img, fx=fator, fy=fator)


# =====================================================================
# ZOOM OUT – Exclusão e Valor Médio
# =====================================================================

def zoom_out_exclusao(img: np.ndarray, fator: float) -> np.ndarray:
    """
    Zoom OUT por exclusão de pixels (subamostragem direta).
    Mantém apenas 1 de cada 'passo' pixels.
    """
    passo = max(1, int(round(fator)))
    return img[::passo, ::passo, :].copy()


def zoom_out_valor_medio(img: np.ndarray, fator: float) -> np.ndarray:
    """
    Zoom OUT por valor médio (media pooling em janelas).
    Para cada bloco de (passo × passo) pixels calcula a média.
    """
    passo = max(1, int(round(fator)))
    h, w = img.shape[:2]

    new_h = h // passo
    new_w = w // passo

    # Recorta para múltiplos exatos do passo
    crop = img[:new_h * passo, :new_w * passo, :]

    # Reshape para blocos e calcula média
    result = (crop
              .reshape(new_h, passo, new_w, passo, img.shape[2])
              .mean(axis=(1, 3)))

    return result.astype(np.uint8)


# =====================================================================
# TRANSFORMAÇÕES COMPOSTAS
# =====================================================================

# Mapa de funções disponíveis
_OP_MAP = {
    "translacao"          : lambda img, p: translacao(img, p.get("tx", 0), p.get("ty", 0)),
    "rotacao"             : lambda img, p: rotacao(img, p.get("angulo", 0)),
    "escala"              : lambda img, p: escala(img, p.get("fx", 1.0), p.get("fy", 1.0)),
    "reflexao"            : lambda img, p: reflexao(img, p.get("eixo", "horizontal")),
    "cisalhamento"        : lambda img, p: cisalhamento(img, p.get("shx", 0.0), p.get("shy", 0.0)),
    "zoom_in_replicacao"  : lambda img, p: zoom_in_replicacao(img, p.get("fator", 2.0)),
    "zoom_in_interpolacao": lambda img, p: zoom_in_interpolacao(img, p.get("fator", 2.0)),
    "zoom_out_exclusao"   : lambda img, p: zoom_out_exclusao(img, p.get("fator", 2.0)),
    "zoom_out_valor_medio": lambda img, p: zoom_out_valor_medio(img, p.get("fator", 2.0)),
}


def composicao_transformacoes(img: np.ndarray, transformacoes: list) -> np.ndarray:
    """
    Aplica uma lista ordenada de transformações sobre a imagem.

    transformacoes é uma lista de dicts:
        [
            {'operacao': 'rotacao',    'params': {'angulo': 45}},
            {'operacao': 'translacao', 'params': {'tx': 30, 'ty': 10}},
            ...
        ]
    """
    resultado = img.copy()
    for etapa in transformacoes:
        nome = etapa.get("operacao", "")
        params = etapa.get("params", {})
        if nome not in _OP_MAP:
            raise ValueError(f"Operação '{nome}' não reconhecida.")
        resultado = _OP_MAP[nome](resultado, params)
    return resultado


def operacoes_disponiveis() -> list:
    return list(_OP_MAP.keys())
