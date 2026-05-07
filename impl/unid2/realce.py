import numpy as np
from PIL import Image

########################################################################
#                          TRANFORMAÇÕES LINEARES                      #
########################################################################

def transformacao_linear_intervalo(img_array: np.ndarray, f_min: int, f_max: int, g_min: int, g_max: int) -> np.ndarray:
    """
    Implementação A: Mapeamento de intervalo [f_min, f_max] para [g_min, g_max].
    """

    # 1. Trabalhar com floats evita o clássicJo problema de "estouro" (overflow)
    # ou perda de decimais durante a divisão na matriz do NumPy.
    f = img_array.astype(np.float64)

    # 2. Proteção contra divisão por zero (caso o usuário coloque f_min igual ao f_max)
    if f_max == f_min:
        raise ValueError("f_max e f_min não podem ser iguais (divisão por zero).")
    
    # 3. A Aplicação da Fórmula
    # g = [(g_max - g_min) / (f_max - f_min)] * (f - f_min) + g_min
    
    taxa_a = (g_max - g_min) / (f_max - f_min)
    g = taxa_a * (f - f_min) + g_min

    # 4. Ajuste de Limites (Clipping)
    # A fórmula matemática pura pode jogar valores para baixo de 0 ou acima de 255
    # O np.clip "corta" tudo que for < 0 para virar 0, e > 255 para virar 255.
    g_clip = np.clip(g, g_min, g_max)
    
    # 5. Conversão de volta para o tipo original (int8)
    g_final = g_clip.astype(np.uint8)
    
    return g_final

def transformacao_linear_por_partes(img_array: np.ndarray, lista_partes: list) -> np.ndarray:
    """
    Implementação B (Fiel ao enunciado): 
    O usuário define N partes. Para cada parte, define o mapeamento [f_min, f_max] -> [g_min, g_max].
    
    lista_partes deve ser uma lista de dicionários. Exemplo:
    [
        {'f_min': 0,   'f_max': 127, 'g_min': 0,  'g_max': 50},  # Parte 1 (Comprime sombras)
        {'f_min': 128, 'f_max': 255, 'g_min': 51, 'g_max': 255}  # Parte 2 (Estica destaques)
    ]
    """
    f = img_array.astype(np.float64)
    
    # Começamos com uma imagem zerada que será preenchida parte a parte
    g = np.zeros_like(f)
    
    # Iteramos sobre a quantidade de partes escolhidas pelo usuário
    for parte in lista_partes:
        f_min, f_max = parte['f_min'], parte['f_max']
        g_min, g_max = parte['g_min'], parte['g_max']
        
        # Cria a máscara: seleciona apenas os pixels que pertencem ao intervalo desta parte específica
        mask = (f >= f_min) & (f <= f_max)
        
        # Aplica a fórmula exata da Transformação A apenas nos pixels da máscara
        if f_max != f_min:
            taxa_a = (g_max - g_min) / (f_max - f_min)
            # Calcula o valor apenas onde a máscara é verdadeira
            g[mask] = taxa_a * (f[mask] - f_min) + g_min
        else:
            g[mask] = g_min  # Proteção contra divisão por zero
            
    # Garantia final para que nenhum cálculo exceda os limites da imagem digital
    g_clip = np.clip(g, 0, 255)
    
    return g_clip.astype(np.uint8)

def transformacao_inversa(img_array: np.ndarray) -> np.ndarray:
    """
    Implementação C: Transformação Inversa (Negativo).
    """
    # O negativo de uma imagem é simplesmente 255 - valor do pixel
    g = 255 - img_array
    return g.astype(np.uint8)

def transformacao_binaria(img_array: np.ndarray, limiar: int) -> np.ndarray:
    """
    Implementação D: Transformação Binária (Thresholding).
    
    limiar: O valor de corte para decidir se o pixel vira preto ou branco.
    """
    # np.where é uma forma eficiente de aplicar a condição em toda a matriz
    g = np.where(img_array < limiar, 0, 255)
    return g.astype(np.uint8)


########################################################################
#                      TRANFORMAÇÕES NÃO LINEARES                      #
########################################################################

def transformacao_logaritmica(img_array: np.ndarray) -> np.ndarray:
    """
    Expande os tons escuros (sombras) e comprime os tons claros.
    Ideal para revelar detalhes em imagens muito escuras.
    """
    f = img_array.astype(np.float64)
    
    # Constante para garantir que o log de 255 não passe de 255 na saída
    c = 255.0 / np.log(1 + 255.0)
    
    g = c * np.log(1 + f)
    return np.clip(g, 0, 255).astype(np.uint8)


def transformacao_raiz(img_array: np.ndarray) -> np.ndarray:
    """
    Similar à logarítmica, mas com uma curva mais suave.
    Clareia a imagem expandindo levemente as sombras.
    """
    f = img_array.astype(np.float64)
    
    c = 255.0 / np.sqrt(255.0)
    
    g = c * np.sqrt(f)
    return np.clip(g, 0, 255).astype(np.uint8)


def transformacao_exponencial(img_array: np.ndarray) -> np.ndarray:
    """
    Comprime os tons escuros e médios, e expande os tons claros.
    Ideal para escurecer imagens "lavadas" ou super expostas.
    """
    f = img_array.astype(np.float64)
    
    # Para evitar overflow, dimensionamos f para estar na mesma escala do logaritmo inverso
    # Isso simula perfeitamente a curva que escurece os tons médios e preserva destaques
    a = np.log(256.0) / 255.0
    c = 255.0 / (np.exp(a * 255.0) - 1.0)
    
    g = c * (np.exp(a * f) - 1.0)
    return np.clip(g, 0, 255).astype(np.uint8)


def transformacao_quadrado(img_array: np.ndarray) -> np.ndarray:
    """
    Similar à exponencial, escurece a imagem globalmente,
    preservando apenas os reflexos e pixels mais claros.
    """
    f = img_array.astype(np.float64)
    
    c = 255.0 / (255.0 ** 2)
    
    g = c * (f ** 2)
    return np.clip(g, 0, 255).astype(np.uint8)

###########################################################################################

def equalizar_histograma(img_array: np.ndarray) -> np.ndarray:
    """
    Realça o contraste da imagem distribuindo uniformemente as intensidades dos pixels.
    """
    # 1. Calcula o Histograma original da imagem
    # np.histogram retorna as contagens e os limites das barras (bins). 
    # O flatten() transforma a matriz 2D numa linha só para contar os pixels.
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    
    # 2. Calcula a CDF (Cumulative Distribution Function / Soma Acumulada)
    cdf = hist.cumsum()
    
    # 3. Normaliza a CDF matemática para o padrão 0 a 255
    # Usamos uma máscara para ignorar os valores que são zero e não estragar a matemática
    cdf_mascarada = np.ma.masked_equal(cdf, 0)
    
    # Aplica a equação de equalização: (CDF - Min) / (Max - Min) * 255
    cdf_mascarada = (cdf_mascarada - cdf_mascarada.min()) * 255 / (cdf_mascarada.max() - cdf_mascarada.min())
    
    # Preenche os buracos da máscara com 0 e converte os valores para inteiros de 8 bits
    cdf_final = np.ma.filled(cdf_mascarada, 0).astype(np.uint8)
    
    # 4. O Remapeamento Mágico do NumPy!
    # A CDF virou um dicionário. Se o pixel original valia 50, o NumPy olha a posição 50 da CDF 
    # e injeta o novo valor diretamente em toda a matriz de uma vez só.
    img_equalizada = cdf_final[img_array]
    
    return img_equalizada

def correcao_gama(img_array: np.ndarray, gama: float) -> np.ndarray:
    """
    Implementação 3: Ajuste de brilho pela Correção Gama.
    Fórmula: g = c * (f ^ gama)
    """
    # 1. Normaliza para o intervalo [0.0, 1.0] para garantir estabilidade matemática
    # Qualquer número entre 0 e 1 elevado a uma potência continua entre 0 e 1!
    f_norm = img_array.astype(np.float64) / 255.0
    
    # 2. Aplica a potência gama (o c nesse caso vira 1, pois f_max = 1.0)
    g_norm = np.power(f_norm, gama)
    
    # 3. Desnormaliza de volta para o espectro de 8 bits [0, 255]
    g = g_norm * 255.0
    
    return np.clip(g, 0, 255).astype(np.uint8)