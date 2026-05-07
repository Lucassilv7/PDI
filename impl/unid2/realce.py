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