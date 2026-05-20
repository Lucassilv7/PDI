import numpy as np
from collections import deque
import heapq
from scipy.ndimage import minimum_filter, label as nd_label


# ─────────────────────────────────────────────
#  UTILITÁRIOS
# ─────────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    """Converte RGB para escala de cinza (luminância)."""
    if img.ndim == 2:
        return img.astype(np.float64)
    
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)
    return 0.299 * R + 0.587 * G + 0.114 * B

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normaliza a imagem para o intervalo [0, 255] e converte para uint8."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = (img - img_min) / (img_max - img_min) * 255
    return normalized.astype(np.uint8)

def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Converte imagem cinza 2D uint8 para RGB 3 canais."""
    gray_uint8 = gray.astype(np.uint8)
    return np.dstack((gray_uint8, gray_uint8, gray_uint8))

def _convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolução 2D manual com padding zero.
    Suporta kernels 2×2 e 3×3.
    Retorna array float64 do mesmo tamanho da entrada.
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    height, width = gray.shape

    padded = np.pad(gray, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    result = np.zeros_like(gray, dtype=np.float64)

    for i in range(kernel_height):
        for j in range(kernel_width):
            result += kernel[i, j] * padded[i:i+height, j:j+width]

    return result

# ─────────────────────────────────────────────
#  1. DETECÇÃO DE PONTOS
# ─────────────────────────────────────────────

KERNEL_POINTS = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float64)

def points_detection(img: np.ndarray, T: int) -> np.ndarray:
    """
    Aplica a máscara laplaciana 3×3 e marca como branco
    os pixels cuja resposta |R| > T.
    Retorna imagem binária RGB.
    """
    gray = to_gray(img)
    R = _convolve2d(gray, KERNEL_POINTS)
    result = np.where(np.abs(R) > T, 255, 0).astype(np.uint8)
    return gray_to_rgb(result)

# ─────────────────────────────────────────────
#  2. DETECÇÃO DE RETAS
# ─────────────────────────────────────────────

KERNEL_LINES ={
    "horizontal": np.array([[-1, -1, -1],
                            [ 2,  2,  2],
                            [-1, -1, -1]], dtype=np.float64),

    "vertical": np.array([[-1, 2, -1],
                          [-1, 2, -1],
                          [-1, 2, -1]], dtype=np.float64),

    "45graus": np.array([[-1, -1,  2],
                         [-1,  2, -1],
                         [ 2, -1, -1]], dtype=np.float64),

    "135graus": np.array([[ 2, -1, -1],
                          [-1,  2, -1],
                          [-1, -1,  2]], dtype=np.float64) 
}

def lines_detection(img: np.ndarray, direction: str) -> np.ndarray:
    """
    Aplica a máscara de detecção de retas na direção especificada.
    direcao: 'horizontal' | 'vertical' | '45graus' | '135graus'
    Retorna imagem normalizada em RGB.
    """
    if direction not in KERNEL_LINES:
        raise ValueError(f"Direção '{direction}' inválida.")
    gray = to_gray(img)
    R = _convolve2d(gray, KERNEL_LINES[direction])
    return gray_to_rgb(normalize_to_uint8(np.abs(R)))

# ─────────────────────────────────────────────
#  3. DETECÇÃO DE BORDAS
# ─────────────────────────────────────────────

def _roberts(gray: np.ndarray) -> np.ndarray:
    
    Kernel_x = np.array([[1,  0],
                         [0, -1]], dtype=np.float64)
    Kernel_y = np.array([[ 0, 1],
                         [-1, 0]], dtype=np.float64)
    Gx = _convolve2d(gray, Kernel_x)
    Gy = _convolve2d(gray, Kernel_y)
    return Gx, Gy

def _roberts_crossed(gray: np.ndarray) -> np.ndarray:
    Kernel_x = np.array([[ 1, 0],
                         [-1, 0]], dtype=np.float64)
    Kernel_y = np.array([[1, -1],
                         [0,  0]], dtype=np.float64)
    Gx = _convolve2d(gray, Kernel_x)
    Gy = _convolve2d(gray, Kernel_y)
    return Gx, Gy

# ── Prewitt ───────────────────────────────────

PREWITT_KERNEL_X = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float64)

PREWITT_KERNEL_Y = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]], dtype=np.float64)

# ── Sobel ─────────────────────────────────────

SOBEL_KERNEL_X = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

SOBEL_KERNEL_Y = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

# ── Kirsch (8 máscaras) ───────────────────────

KIRSCH_KERNELS = [
    np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float64),
    np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float64),
    np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float64),
    np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float64),
    np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float64),
    np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float64),
    np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float64),
    np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float64),
]

# ── Robinson (8 máscaras) ─────────────────────

ROBINSON_KERNELS = [
    np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]], dtype=np.float64),
    np.array([[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]], dtype=np.float64),
    np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64),
    np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]], dtype=np.float64),
    np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], dtype=np.float64),
    np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]], dtype=np.float64),
    np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=np.float64),
    np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]], dtype=np.float64),
]

# ── Frei-Chen (9 máscaras) ───────────────────
_s2 = np.sqrt(2)
FREI_CHEN_MASKS = [
    # M1–M4: bordas
    np.array([[1, _s2, 1], [0, 0, 0], [-1, -_s2, -1]],   dtype=np.float64) / (2 + _s2),
    np.array([[1, 0, -1], [_s2, 0, -_s2], [1, 0, -1]],   dtype=np.float64) / (2 + _s2),
    np.array([[0, -1, _s2], [1, 0, -1], [-_s2, 1, 0]],   dtype=np.float64) / (2 + _s2),
    np.array([[_s2, -1, 0], [-1, 0, 1], [0, 1, -_s2]],   dtype=np.float64) / (2 + _s2),
    # M5–M8: retas
    np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]],         dtype=np.float64) / 2,
    np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]],         dtype=np.float64) / 2,
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]],       dtype=np.float64) / 6,
    np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]],       dtype=np.float64) / 6,
    # M9: média
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]],           dtype=np.float64) / 9,
]

# ── Laplaciano ────────────────────────────────
LAPLACIAN_KERNEL_H1 = np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]], dtype=np.float64)

LAPLACIAN_KERNEL_H2 = np.array([[-1, -4, -1],
                                [-4, 20, -4],
                                [-1, -4, -1]], dtype=np.float64)

def edges_detection(img: np.ndarray, method: str) -> np.ndarray:
    """
    Calcula bordas pelo método especificado.
    metodo: 'roberts' | 'roberts_cruzado' | 'prewitt' | 'sobel' |
            'kirsch' | 'robinson' | 'frei_chen' |
            'laplaciano_h1' | 'laplaciano_h2'
 
    Retorna dict com chaves dependendo do método:
      - 'gx', 'gy', 'magnitude'  → para Roberts, Prewitt, Sobel
      - 'magnitude'              → para Kirsch, Robinson, Frei-Chen, Laplaciano
    Todos os valores são arrays RGB uint8.
    """
    gray = to_gray(img)
    out = {}

    if method in ("roberts", "roberts_cruzado"):
        Gx, Gy = _roberts(gray) if method == "roberts" else _roberts_crossed(gray)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        out['gx']        = gray_to_rgb(normalize_to_uint8(Gx))
        out['gy']        = gray_to_rgb(normalize_to_uint8(Gy))
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))

    elif method == "prewitt":
        Gx = _convolve2d(gray, PREWITT_KERNEL_X)
        Gy = _convolve2d(gray, PREWITT_KERNEL_Y)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        out['gx']        = gray_to_rgb(normalize_to_uint8(Gx))
        out['gy']        = gray_to_rgb(normalize_to_uint8(Gy))
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))

    elif method == "sobel":
        Gx = _convolve2d(gray, SOBEL_KERNEL_X)
        Gy = _convolve2d(gray, SOBEL_KERNEL_Y)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        out['gx']        = gray_to_rgb(normalize_to_uint8(Gx))
        out['gy']        = gray_to_rgb(normalize_to_uint8(Gy))
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))
    
    elif method == "kirsch":
        results = np.stack([_convolve2d(gray, m) for m in KIRSCH_KERNELS], axis=0)
        magnitude = results.max(axis=0)
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))

    elif method == "robinson":
        results = np.stack([_convolve2d(gray, m) for m in ROBINSON_KERNELS], axis=0)
        magnitude = results.max(axis=0)
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))

    elif method == "frei_chen":
        # Calcula pesos w_i = <janela, M_i> para cada máscara
        ws = [_convolve2d(gray, m) for m in FREI_CHEN_MASKS]
        # Soma dos quadrados dos pesos de borda (M1-M4) / soma total
        numerator   = sum(w**2 for w in ws[:4])
        denominator = sum(w**2 for w in ws) + 1e-10  # Evita divisão por zero
        magnitude   = np.sqrt(numerator / denominator)
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(magnitude))

    elif method in ("laplaciano_h1", "laplaciano_h2"):
        kernel = LAPLACIAN_KERNEL_H1 if method == "laplaciano_h1" else LAPLACIAN_KERNEL_H2
        R = _convolve2d(gray, kernel)
        out['magnitude'] = gray_to_rgb(normalize_to_uint8(np.abs(R)))

    else:
        raise ValueError(f"Método de borda'{method}' inválido.")
    
    return out 

# ─────────────────────────────────────────────
#  4. LIMIARIZAÇÃO
# ─────────────────────────────────────────────

def global_limiarization(img: np.ndarray) -> np.ndarray:
    """
    Limiarização global iterativa
    Retorna (imagem_binaria_RGB, T_final).
    """

    gray = to_gray(img)

    T = gray.mean()
    while True:
        group1 = gray[gray <= T]
        group2 = gray[gray > T]
        median1 = group1.mean() if group1.size > 0 else 0.0
        median2 = group2.mean() if group2.size > 0 else 255.0
        new_T = (median1 + median2) / 2.0
        if abs(new_T - T) < 0.5:
            break
        T = new_T

    binary = np.where(gray > T, 255, 0).astype(np.uint8)
    return gray_to_rgb(binary), round(T, 2)

def _base_local_limiarization(gray: np.ndarray, n: int, mode: str, k: float = 0.0) -> np.ndarray:
    """
    Limiarização local genérica.
    modo: 'media' | 'maximo' | 'minimo' | 'niblack'
    """
    height, width = gray.shape
    r = n // 2
    # Padding refletido para evitar artefatos nas bordas
    padded = np.pad(gray, r, mode='reflect')
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            window = padded[i:i+n, j:j+n]
            if mode == 'media':
                T = window.mean()
            elif mode == 'maximo':
                T = window.max()
            elif mode == 'minimo':
                T = window.min()
            elif mode == 'niblack':
                T = window.mean() - k * window.std()
            else:
                T = window.mean()  # Default para evitar erro, mas não deve ocorrer

            binary[i, j] = 255 if gray[i, j] > T else 0

    return binary

def local_limiarization(img: np.ndarray, n: int = 15, mode: str = "media", k: float = -0.2) -> np.ndarray:
    """
    Limiarização local.
    modo: 'media' | 'maximo' | 'minimo' | 'niblack'
    n: tamanho da janela (ímpar recomendado)
    k: parâmetro de Niblack (usado apenas se modo='niblack')
    Retorna imagem binária RGB.
    """
    gray = to_gray(img)
    if n % 2 == 0:
        n += 1  # Garante janela com centro definido
    result = _base_local_limiarization(gray, n, mode, k)
    return gray_to_rgb(result)

# ─────────────────────────────────────────────
#  5A. CRESCIMENTO DE REGIÃO
# ─────────────────────────────────────────────

# Paleta de cores para até 20 regiões (RGB)
_PALETA = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

_VIZINHOS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def region_growing(img: np.ndarray, seeds: list, T: float) -> np.ndarray:
    """
    Segmentação por crescimento de região com BFS.
 
    Parâmetros:
        img     : imagem RGB de entrada
        sementes: lista de tuplas (y, x)  — coordenadas das sementes
        T       : limiar de diferença de intensidade
 
    Retorna imagem pseudocolorida RGB onde cada região tem uma cor distinta
    e pixels não atribuídos ficam pretos.
    """
    gray = to_gray(img)
    height, width = gray.shape
    labels = np.zeros((height, width), dtype=np.int32)  # 0 = sem rótulo

    queeue = deque()

    # Inicializa sementes
    for idx, (sy, sx) in enumerate(seeds):
        label = idx + 1  
        labels[sy, sx] = label
        queeue.append((sy, sx, label, gray[sy, sx]))

    # BFS
    while queeue:
        y, x, label, seed_value = queeue.popleft()
        for dy, dx in _VIZINHOS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and labels[ny, nx] == 0:
                if abs(float(gray[ny, nx]) - float(seed_value)) <= T:
                    labels[ny, nx] = label
                    queeue.append((ny, nx, label, seed_value))
    
    # Pseudocolorização
    results = np.zeros((height, width, 3), dtype=np.uint8)
    for label in range(1, len(seeds) + 1):
        color = _PALETA[(label - 1) % len(_PALETA)]
        mask = labels == label
        results[mask] = color
    
    return results

# ─────────────────────────────────────────────
#  5B. WATERSHED
# ─────────────────────────────────────────────

def _find_local_minima(gradient: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Encontra mínimos locais no mapa de gradiente usando uma janela
    de tamanho (2*min_distance+1)^2.
 
    Retorna máscara booleana onde True = mínimo local confirmado.
 
    Estratégia:
      - Para cada pixel, verifica se é o menor valor em sua vizinhança.
      - Depois aplica supressão de não-máximos para manter apenas um
        mínimo por cluster (evita centenas de mínimos vizinhos no mesmo vale).
    """

    # Mínimo local: pixel igual ao mínimo da sua janela
    footprint_size = 2 * min_distance + 1
    local_min = minimum_filter(gradient, size=footprint_size)
    minima_mask = (gradient == local_min)
 
    # Supressão: dentro de cada grupo conectado de mínimos, mantém só o centróide
    labeled, num = nd_label(minima_mask)
    result = np.zeros_like(minima_mask, dtype=bool)
    for region_id in range(1, num + 1):
        coords = np.argwhere(labeled == region_id)
        cy, cx = coords.mean(axis=0).astype(int)
        result[cy, cx] = True
 
    return result
 
 
def _watershed_core(img: np.ndarray):
    """
    Watershed por imersão (Vincent & Soille, 1991) com marcadores reais.
 
    Pipeline:
      1. Suaviza a imagem (blur 5×5) para atenuar ruído antes do gradiente.
      2. Calcula gradiente Sobel como "relevo topográfico".
      3. Identifica marcadores: pixels cujo gradiente é mínimo local em
         janela min_distance×min_distance — um por vale.
      4. Inunda o relevo a partir dos marcadores em ordem crescente de
         gradiente (fila de prioridade). Apenas vizinhos de pixels já
         rotulados entram na fila — nunca pixels aleatórios.
      5. Pixel vizinho de dois rótulos distintos = linha de contenção.
 
    A diferença crítica em relação à versão anterior:
      - Nunca cria novo rótulo durante a propagação, apenas nos marcadores.
      - Usa FIFO por nível de gradiente (não heap global com todos os pixels).
    """
 
    gray = to_gray(img).astype(np.float64)
    h, w = gray.shape
 
    # 1. Suavização
    kb = np.ones((5, 5), dtype=np.float64) / 25.0
    gray_s = _convolve2d(gray, kb)
 
    # 2. Gradiente
    Gx = _convolve2d(gray_s, SOBEL_KERNEL_X)
    Gy = _convolve2d(gray_s, SOBEL_KERNEL_Y)
    gradient = np.sqrt(Gx**2 + Gy**2)
    g_max = gradient.max() + 1e-10
    grad_int = (gradient / g_max * 255).astype(np.int32)
 
    # 3. Marcadores: mínimos locais em janela 9×9, um por grupo conexo
    win = max(3, min(15, h // 8, w // 8))  # adapta ao tamanho da imagem
    local_min_val = minimum_filter(grad_int, size=win)
    minima_mask = (grad_int == local_min_val)
    labeled_groups, n_groups = nd_label(minima_mask)
 
    WSHED  = -1
    INQUEUE = -2
    labels = np.zeros((h, w), dtype=np.int32)
 
    heap = []
    for gid in range(1, n_groups + 1):
        coords = np.argwhere(labeled_groups == gid)
        # representante: pixel de menor gradiente no grupo
        best = min(coords, key=lambda c: grad_int[c[0], c[1]])
        by, bx = best
        labels[by, bx] = gid
        heapq.heappush(heap, (int(grad_int[by, bx]), int(by), int(bx)))
 
    # 4. Propagação BFS por prioridade — somente vizinhos de rotulados entram
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
 
    while heap:
        _, y, x = heapq.heappop(heap)
        cur_label = labels[y, x]
        if cur_label in (WSHED, INQUEUE):
            continue
 
        for dy, dx in neighbors_4:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            nb_label = labels[ny, nx]
            if nb_label == 0:
                # Ainda sem rótulo: herda do pai
                labels[ny, nx] = cur_label
                heapq.heappush(heap, (int(grad_int[ny, nx]), int(ny), int(nx)))
            elif nb_label > 0 and nb_label != cur_label:
                # Dois rótulos distintos se encontram: linha de contenção
                labels[ny, nx] = WSHED
 
    return labels, WSHED
 
 
def watershed(img: np.ndarray) -> np.ndarray:
    """
    Watershed com marcadores baseados em mínimos locais do gradiente.
    Retorna imagem original com as linhas de contenção desenhadas em preto.
    """
    labels, WSHED = _watershed_core(img)
    result = img.copy()
    result[labels == WSHED] = [0, 0, 0]
    return result
 
 
def watershed_with_markers(img: np.ndarray) -> tuple:
    """
    Retorna:
      - imagem original com linhas de contenção em preto
      - imagem pseudocolorida das regiões encontradas
    """
    labels, WSHED = _watershed_core(img)
    h, w = img.shape[:2]
 
    # Linhas de contenção em preto
    img_lines = img.copy()
    img_lines[labels == WSHED] = [0, 0, 0]
 
    # Pseudocolorização das regiões
    pseudocolor = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(labels)
    unique = unique[unique > 0]  # exclui WSHED (-1) e não-rotulados (0)
    for i, label in enumerate(unique):
        color = _PALETA[i % len(_PALETA)]
        pseudocolor[labels == label] = color
    pseudocolor[labels == WSHED] = [0, 0, 0]
 
    return img_lines, pseudocolor