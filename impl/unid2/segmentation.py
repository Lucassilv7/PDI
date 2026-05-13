import numpy as np
from collections import deque
import heapq


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

def watershed(img: np.ndarray) -> np.ndarray:
    """
    Segmentação por Watershed (algoritmo de imersão simplificado).
 
    1. Calcula gradiente Sobel da imagem em cinza.
    2. Processa pixels em ordem crescente de gradiente (fila de prioridade).
    3. Pixels na fronteira entre regiões distintas são marcados como
       linha de contenção (watershed line) — exibidas em vermelho.
 
    Retorna imagem original com as linhas de contenção desenhadas em vermelho.
    """
    gray = to_gray(img)
    height, width = gray.shape

    # Gradiente (magnitude Sobel normalizada para [0,255])
    Gx = _convolve2d(gray, SOBEL_KERNEL_X)
    Gy = _convolve2d(gray, SOBEL_KERNEL_Y)
    gradient = np.sqrt(Gx**2 + Gy**2)
    gradient_norm = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)

    WATERSHED_LINE = -1
    labels = np.zeros((height, width), dtype=np.int32)  # 0 = sem rótulo
    next_label = 1

    # Fila de prioridade: (gradiente, y, x)
    heap = []
    for y in range(height):
        for x in range(width):
            heapq.heappush(heap, (gradient_norm[y, x], y, x))

    neighbors_4 = [(-1,0),(1,0),(0,-1),(0,1)]

    while heap:
        _, y, x = heapq.heappop(heap)
        if labels[y, x] != 0:
            continue  # Já processado

        # Verifica rótulos dos vizinhos já processados
        neighbor_labels = set()
        for dy, dx in neighbors_4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and labels[ny, nx] > 0:
                neighbor_labels.add(labels[ny, nx])
        
        if len(neighbor_labels) == 0:
            # Novo mínimo local → nova região
            labels[y, x] = next_label
            next_label += 1
        elif len(neighbor_labels) == 1:
            # Único rótulo vizinho → mesmo rótulo
            labels[y, x] = neighbor_labels.pop()
        else:
            # Fronteira entre regiões → linha de contenção
            labels[y, x] = WATERSHED_LINE

    # Desenha linhas de contenção em vermelho sobre a imagem original
    results = img.copy()
    line_mask = labels == WATERSHED_LINE
    results[line_mask] = [255, 0, 0] 

    return results

def watershed_with_markers(img: np.ndarray) -> np.ndarray:
    """
    Retorna a imagem com linhas de contenção E a imagem pseudocolorida
    das regiões encontradas.
    """
    gray = to_gray(img)
    height, width = gray.shape

    Gx = _convolve2d(gray, SOBEL_KERNEL_X)
    Gy = _convolve2d(gray, SOBEL_KERNEL_Y)
    gradient = np.sqrt(Gx**2 + Gy**2)
    gradient_norm = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)

    WATERSHED_LINE = -1
    labels = np.zeros((height, width), dtype=np.int32)  
    next_label = 1

    heap = []
    for y in range(height):
        for x in range(width):
            heapq.heappush(heap, (gradient_norm[y, x], y, x))

    neighbors_4 = [(-1,0),(1,0),(0,-1),(0,1)]

    while heap:
        _, y, x = heapq.heappop(heap)
        if labels[y, x] != 0:
            continue  

        neighbor_labels = set()
        for dy, dx in neighbors_4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and labels[ny, nx] > 0:
                neighbor_labels.add(labels[ny, nx])
        
        if len(neighbor_labels) == 0:
            labels[y, x] = next_label
            next_label += 1
        elif len(neighbor_labels) == 1:
            labels[y, x] = neighbor_labels.pop()
        else:
            labels[y, x] = WATERSHED_LINE

    results = img.copy()
    line_mask = labels == WATERSHED_LINE
    results[line_mask] = [255, 0, 0] 

    # Imagem pseudocolorida das regiões
    pseudocolor = np.zeros((height, width, 3), dtype=np.uint8)
    n_labels = next_label - 1
    for label in range(1, n_labels + 1):
        color = _PALETA[(label - 1) % len(_PALETA)]
        mask = labels == label
        pseudocolor[mask] = color
    
    pseudocolor[labels == WATERSHED_LINE] = [255, 0, 0]

    return results, pseudocolor