import numpy as np
from PIL import Image

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