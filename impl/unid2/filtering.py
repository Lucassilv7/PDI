import numpy as np
from scipy import stats
from scipy.signal import convolve2d

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

def _aplicar_filtro_variancia(img_array: np.ndarray, mascaras: list) -> np.ndarray:
    """
    Função base que desliza uma janela 5x5 pela imagem.
    Para cada pixel, testa todas as 'mascaras' recebidas, calcula a variância
    de cada sub-região e atribui ao pixel central a média da região com menor variância.
    """
    # A janela para estes filtros clássicos é sempre 5x5, logo o padding é 2
    pad_width = 2 
    img_padded = np.pad(img_array, pad_width, mode='edge')
    
    linhas, colunas = img_array.shape
    img_filtrada = np.zeros_like(img_array, dtype=np.float64)
    
    # Desliza a janela 5x5 por toda a imagem
    for i in range(linhas):
        for j in range(colunas):
            vizinhanca = img_padded[i : i + 5, j : j + 5]
            
            menor_variancia = float('inf')
            melhor_media = 0.0
            
            # Avalia cada sub-região definida pelo algoritmo escolhido
            for mascara in mascaras:
                # Extrai apenas os pixels pertencentes a esta máscara
                pixels_regiao = vizinhanca[mascara]
                
                var = np.var(pixels_regiao)
                
                # Guarda a média da região mais homogénea encontrada
                if var < menor_variancia:
                    menor_variancia = var
                    melhor_media = np.mean(pixels_regiao)
            
            img_filtrada[i, j] = melhor_media
            
    return np.clip(img_filtrada, 0, 255).astype(np.uint8)


# =========================================================================
# FILTROS PASSA-BAIXA (PRESERVAÇÃO DE BORDAS)
# =========================================================================

def filtro_kuwahara(img_array: np.ndarray) -> np.ndarray:
    """
    Filtro de Kuwahara (ou Kawahara).
    Divide a janela 5x5 em 4 regiões quadradas de 3x3 nos cantos.
    """
    mascaras = []
    # Região 1: Canto Superior Esquerdo
    m1 = np.zeros((5, 5), dtype=bool); m1[0:3, 0:3] = True
    mascaras.append(m1)
    # Região 2: Canto Superior Direito
    m2 = np.zeros((5, 5), dtype=bool); m2[0:3, 2:5] = True
    mascaras.append(m2)
    # Região 3: Canto Inferior Esquerdo
    m3 = np.zeros((5, 5), dtype=bool); m3[2:5, 0:3] = True
    mascaras.append(m3)
    # Região 4: Canto Inferior Direito
    m4 = np.zeros((5, 5), dtype=bool); m4[2:5, 2:5] = True
    mascaras.append(m4)
    
    return _aplicar_filtro_variancia(img_array, mascaras)

def filtro_tomita_tsuji(img_array: np.ndarray) -> np.ndarray:
    """
    Filtro de Tomita e Tsuji.
    Usa as mesmas 4 regiões do Kuwahara + 1 região 3x3 perfeitamente no centro.
    """
    # Recria as 4 máscaras do Kuwahara para não duplicar código visualmente
    m1 = np.zeros((5, 5), dtype=bool); m1[0:3, 0:3] = True
    m2 = np.zeros((5, 5), dtype=bool); m2[0:3, 2:5] = True
    m3 = np.zeros((5, 5), dtype=bool); m3[2:5, 0:3] = True
    m4 = np.zeros((5, 5), dtype=bool); m4[2:5, 2:5] = True
    mascaras = [m1, m2, m3, m4]
    
    # Adiciona a 5ª máscara (Centro)
    m5 = np.zeros((5, 5), dtype=bool); m5[1:4, 1:4] = True
    mascaras.append(m5)
    
    return _aplicar_filtro_variancia(img_array, mascaras)

def filtro_nagao_matsuyama(img_array: np.ndarray) -> np.ndarray:
    """
    Filtro de Nagao e Matsuyama.
    Usa 9 regiões poligonais (formato de "casas") contendo 7 pixels cada, 
    abrangendo as laterais, cantos e o centro.
    """
    coords = [
        [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)], # Centro (9 pixels)
        [(0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2)], # Topo
        [(4,1), (4,2), (4,3), (3,1), (3,2), (3,3), (2,2)], # Baixo
        [(1,0), (2,0), (3,0), (1,1), (2,1), (3,1), (2,2)], # Esquerda
        [(1,4), (2,4), (3,4), (1,3), (2,3), (3,3), (2,2)], # Direita
        [(0,0), (0,1), (1,0), (1,1), (0,2), (2,0), (2,2)], # Canto Superior Esquerdo
        [(0,4), (0,3), (1,4), (1,3), (0,2), (2,4), (2,2)], # Canto Superior Direito
        [(4,0), (4,1), (3,0), (3,1), (4,2), (2,0), (2,2)], # Canto Inferior Esquerdo
        [(4,4), (4,3), (3,4), (3,3), (4,2), (2,4), (2,2)]  # Canto Inferior Direito
    ]
    
    mascaras = []
    for coord_list in coords:
        m = np.zeros((5, 5), dtype=bool)
        for y, x in coord_list: m[y, x] = True
        mascaras.append(m)
        
    return _aplicar_filtro_variancia(img_array, mascaras)

def filtro_somboonkaew(img_array: np.ndarray) -> np.ndarray:
    """
    Filtro de Somboonkaew.
    Usa 8 regiões direcionais (fatias) partindo do centro apontando para as extremidades.
    """
    coords = [
        [(2,2), (1,2), (0,2), (0,1), (0,3)], # Norte
        [(2,2), (3,2), (4,2), (4,1), (4,3)], # Sul
        [(2,2), (2,1), (2,0), (1,0), (3,0)], # Oeste
        [(2,2), (2,3), (2,4), (1,4), (3,4)], # Leste
        [(2,2), (1,1), (0,0), (0,1), (1,0)], # Noroeste
        [(2,2), (1,3), (0,4), (0,3), (1,4)], # Nordeste
        [(2,2), (3,1), (4,0), (4,1), (3,0)], # Sudoeste
        [(2,2), (3,3), (4,4), (4,3), (3,4)]  # Sudeste
    ]
    
    mascaras = []
    for coord_list in coords:
        m = np.zeros((5, 5), dtype=bool)
        for y, x in coord_list: m[y, x] = True
        mascaras.append(m)
        
    return _aplicar_filtro_variancia(img_array, mascaras)

# =========================================================================
# FILTROS PASSA-ALTA (REALCE DE DETALHES)
# =========================================================================

def obter_mascara_passa_alta(tipo: str) -> np.ndarray:
    """
    Retorna o kernel 3x3 correspondente ao filtro passa-alta solicitado.
    (Nota: Se os slides do professor tiverem números diferentes para M1, M2 ou M3, 
    basta alterar os números nestas matrizes!)
    """
    mascaras = {
        'H1': np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ]),
        
        'H2': np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]),
        
        # M1, M2 e M3 costumam ser variações direcionais ou de ênfase
        'M1': np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]), # Deteção de linhas horizontais
        
        'M2': np.array([
            [1, -2,  1],
            [-2, 5, -2],
            [1,  -2, 1]
        ]), # Deteção de linhas verticais
        
        'M3': np.array([
            [0, -1, 0],
            [-1,  5, -1],
            [0, -1, 0]
        ])  # Deteção de linhas diagonais
    }
    
    tipo = tipo.upper()
    if tipo not in mascaras:
        raise ValueError(f"Máscara {tipo} desconhecida.")
        
    return mascaras[tipo]


def aplicar_filtro_passa_alta(img_array: np.ndarray, tipo: str) -> np.ndarray:
    """
    Aplica a convolução 2D de um filtro passa-alta sobre a imagem.
    """
    # 1. Obter a máscara desejada
    kernel = obter_mascara_passa_alta(tipo)
    
    # 2. Garantir que a imagem está num formato que suporte números negativos
    img_float = img_array.astype(np.float64)
    
    # 3. Aplicar a Convolução Verdadeira
    # mode='same' garante que a imagem de saída tem o mesmo tamanho da original
    # boundary='symm' resolve o problema das bordas espelhando os píxeis
    img_conv = convolve2d(img_float, kernel, mode='same', boundary='symm')
    
    # 4. Tratar os valores negativos gerados pelas transições de borda
    img_abs = np.abs(img_conv)
    
    # 5. Normalizar/Cortar para o intervalo visível (0-255)
    img_final = np.clip(img_abs, 0, 255).astype(np.uint8)
    
    return img_final