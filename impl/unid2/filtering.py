import numpy as np
from scipy import stats

def _deslizar_janela(img_array: np.ndarray, kernel_size: int, operacao) -> np.ndarray:
    """
    Função base que cuida do padding e de deslizar a janela (NxN) sobre a imagem.
    Recebe uma 'operacao' (função do numpy/scipy) que será aplicada em cada vizinhança.
    """
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho da janela deve ser ímpar (3, 5, 7...).")
    
    pad_width = kernel_size // 2
    # Preenche as bordas repetindo o pixel da ponta (evita bordas pretas)
    img_padded = np.pad(img_array, pad_width, mode='edge')

    linhas, colunas = img_array.shape
    img_filtrada = np.zeros_like(img_array, dtype=np.float64)

    for i in range(linhas):
        for j in range(colunas):
            vizinhanca = img_padded[i : i + kernel_size, j : j + kernel_size]
            img_filtrada[i, j] = operacao(vizinhanca)
            
    return np.clip(img_filtrada, 0, 255).astype(np.uint8)

# =========================================================================
# FILTROS PASSA-BAIXA (SUAVIZAÇÃO E MORFOLOGIA)
# =========================================================================

def filtro_media(img_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Filtro da Média (Borramento).
    Soma todos os pixels da janela e divide pelo total.
    """
    return _deslizar_janela(img_array, kernel_size, np.mean)

def filtro_mediana(img_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Filtro da Mediana.
    Ordena os pixels da janela e pega o valor do meio.
    Muito eficaz para remover ruído "sal e pimenta".
    """
    return _deslizar_janela(img_array, kernel_size, np.median)

def filtro_maximo(img_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Filtro de Máximo (Dilatação).
    Escolhe o pixel mais claro da janela, expandindo áreas claras.
    """
    return _deslizar_janela(img_array, kernel_size, np.max)

def filtro_minimo(img_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Filtro de Mínimo (Erosão).
    Escolhe o pixel mais escuro da janela, expandindo áreas escuras.
    """
    return _deslizar_janela(img_array, kernel_size, np.min)

def filtro_moda(img_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Filtro da Moda.
    Escolhe a intensidade de pixel que mais se repete na janela.
    """
    # Como o stats.mode retorna um objeto complexo e não apenas o número,
    # criamos uma função lambda rápida (wrapper) para extrair apenas o valor desejado.
    def calc_moda(vizinhanca):
        resultado = stats.mode(vizinhanca, axis=None, keepdims=False)
        return resultado.mode
        
    return _deslizar_janela(img_array, kernel_size, calc_moda)