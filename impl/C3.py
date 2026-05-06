import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image

def image_to_array(img_src: str) -> np.ndarray:
    """ Converte uma imagem PIL para um array NumPy. """
    img = Image.open(img_src).convert("RGB")
    return np.array(img)

# =====================================================================
# 1. DECOMPOSIÇÃO MONOCROMÁTICA (Implementação 3A)
# =====================================================================

def decouple_rgb(img_array: np.ndarray) -> tuple:
    """ Separa a imagem RGB nos canais R00, 0G0 e 00B. """
    r_array = np.zeros_like(img_array)
    g_array = np.zeros_like(img_array)
    b_array = np.zeros_like(img_array)

    r_array[:, :, 0] = img_array[:, :, 0]
    g_array[:, :, 1] = img_array[:, :, 1]
    b_array[:, :, 2] = img_array[:, :, 2]

    recoupled_array = r_array + g_array + b_array

    return r_array, g_array, b_array, recoupled_array


# =====================================================================
# 2. CONVERSÕES MATEMÁTICAS BASE (Implementação 3B)
# =====================================================================

def rgb_to_cmy(img_array: np.ndarray) -> tuple:
    """ Função matemática pura: Converte RGB para CMY """
    # Normaliza a imagem para 0.0 - 1.0
    rgb_norm = img_array.astype(np.float64) / 255.0

    R = rgb_norm[:, :, 0]
    G = rgb_norm[:, :, 1]
    B = rgb_norm[:, :, 2]

    # Inverso aditivo (Fórmula do CMY)
    C = 1.0 - R
    M = 1.0 - G
    Y = 1.0 - B

    return C, M, Y

def rgb_to_cmyk(img_array: np.ndarray) -> tuple:
    rgb_norm = img_array.astype(np.float64) / 255.0
    R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
    
    C = 1.0 - R
    M = 1.0 - G
    Y = 1.0 - B
    
    K = np.minimum(np.minimum(C, M), Y)
    mask = (K < 1.0)
    
    C_k = np.zeros_like(C)
    M_k = np.zeros_like(M)
    Y_k = np.zeros_like(Y)
    
    C_k[mask] = (C[mask] - K[mask]) / (1.0 - K[mask])
    M_k[mask] = (M[mask] - K[mask]) / (1.0 - K[mask])
    Y_k[mask] = (Y[mask] - K[mask]) / (1.0 - K[mask])
    
    return C_k, M_k, Y_k, K

def rgb_to_yuv(img_array: np.ndarray) -> tuple:
    R = img_array[:, :, 0].astype(np.float64) / 255.0
    G = img_array[:, :, 1].astype(np.float64) / 255.0
    B = img_array[:, :, 2].astype(np.float64) / 255.0
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.312 * B
    
    return Y, U, V

def rgb_to_hsv(img_array: np.ndarray) -> tuple:
    rgb_norm = img_array.astype(np.float64) / 255.0
    R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
    
    Cmax = np.max(rgb_norm, axis=2)
    Cmin = np.min(rgb_norm, axis=2)
    delta = Cmax - Cmin
    
    V = Cmax
    
    S = np.zeros_like(Cmax)
    mask_s = Cmax > 0
    S[mask_s] = delta[mask_s] / Cmax[mask_s]
    
    H = np.zeros_like(Cmax)
    mask_delta = delta > 0
    
    mask_R = mask_delta & (Cmax == R)
    mask_G = mask_delta & (Cmax == G)
    mask_B = mask_delta & (Cmax == B)
    
    H[mask_R] = 60.0 * (((G[mask_R] - B[mask_R]) / delta[mask_R]) % 6.0)
    H[mask_G] = 60.0 * (((B[mask_G] - R[mask_G]) / delta[mask_G]) + 2.0)
    H[mask_B] = 60.0 * (((R[mask_B] - G[mask_B]) / delta[mask_B]) + 4.0)

    return H, S, V


# =====================================================================
# 3. GERADORES DE IMAGENS TINTADAS (Para Exibição Visual)
# =====================================================================

def get_cmy_tinted(img_array: np.ndarray) -> tuple:
    """ Retorna as matrizes colorizadas do CMY prontas para o Tkinter """
    C, M, Y = rgb_to_cmy(img_array)
    altura, largura = C.shape
    
    # Simula o Ciano (Absorve o Red)
    cyan_img = np.ones((altura, largura, 3))
    cyan_img[:, :, 0] = 1.0 - C 
    
    # Simula o Magenta (Absorve o Green)
    magenta_img = np.ones((altura, largura, 3))
    magenta_img[:, :, 1] = 1.0 - M 
    
    # Simula o Amarelo (Absorve o Blue)
    yellow_img = np.ones((altura, largura, 3))
    yellow_img[:, :, 2] = 1.0 - Y

    # Converte de volta para a escala 0-255 (formato de imagem)
    c_uint8 = (cyan_img * 255).astype(np.uint8)
    m_uint8 = (magenta_img * 255).astype(np.uint8)
    y_uint8 = (yellow_img * 255).astype(np.uint8)

    return c_uint8, m_uint8, y_uint8

def get_cmyk_tinted_images(img_array: np.ndarray) -> tuple:
    C, M, Y, K = rgb_to_cmyk(img_array)
    altura, largura = C.shape
    
    cyan_img = np.ones((altura, largura, 3))
    cyan_img[:, :, 0] = 1.0 - C 
    
    magenta_img = np.ones((altura, largura, 3))
    magenta_img[:, :, 1] = 1.0 - M 
    
    yellow_img = np.ones((altura, largura, 3))
    yellow_img[:, :, 2] = 1.0 - Y
    
    black_img = np.ones((altura, largura, 3))
    black_img[:, :, 0] = black_img[:, :, 1] = black_img[:, :, 2] = 1.0 - K

    return (cyan_img * 255).astype(np.uint8), \
           (magenta_img * 255).astype(np.uint8), \
           (yellow_img * 255).astype(np.uint8), \
           (black_img * 255).astype(np.uint8)

def get_yuv_tinted_images(img_array: np.ndarray) -> tuple:
    Y, U, V = rgb_to_yuv(img_array)
    
    def yuv_to_rgb(y, u, v):
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return np.clip(np.dstack((r, g, b)), 0, 1)

    zero_uv = np.zeros_like(Y)
    mid_y = np.full_like(Y, 0.5)

    img_y = yuv_to_rgb(Y, zero_uv, zero_uv)
    img_u = yuv_to_rgb(mid_y, U, zero_uv)
    img_v = yuv_to_rgb(mid_y, zero_uv, V)

    return (img_y * 255).astype(np.uint8), \
           (img_u * 255).astype(np.uint8), \
           (img_v * 255).astype(np.uint8)

def get_hsv_clean_images(img_array: np.ndarray) -> tuple:
    H, S, V_hsv = rgb_to_hsv(img_array)
    
    H_norm = H / 360.0 
    ones = np.ones_like(H)
    
    # Matiz limpo preservando as sombras originais
    img_h_limpo = hsv_to_rgb(np.dstack((H_norm, ones, V_hsv)))
    
    # Saturação e Valor replicados nos 3 canais para virarem imagens em tons de cinza
    img_s = np.dstack((S, S, S))
    img_v = np.dstack((V_hsv, V_hsv, V_hsv))
    
    return (img_h_limpo * 255).astype(np.uint8), \
           (img_s * 255).astype(np.uint8), \
           (img_v * 255).astype(np.uint8)


# =====================================================================
# 4. PSEUDOCOLORIZAÇÃO E REDISTRIBUIÇÃO (Implementações 3C e 3D)
# =====================================================================

def density_slicing(img_array: np.ndarray) -> tuple:
    """ Aplica fatiamento por densidade. Retorna a imagem em tons de cinza base e a pseudo-imagem. """
    
    # Extrai luminância (tons de cinza) diretamente da matriz RGB para não depender do PIL aqui
    R = img_array[:, :, 0].astype(np.float64)
    G = img_array[:, :, 1].astype(np.float64)
    B = img_array[:, :, 2].astype(np.float64)
    gray_array = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    altura, largura = gray_array.shape
    pseudo_img = np.zeros((altura, largura, 3), dtype=np.uint8)

    # Fatias
    mask_1 = (gray_array >= 0) & (gray_array <= 60)
    pseudo_img[mask_1] = [160, 57, 0]

    mask_2 = (gray_array >= 61) & (gray_array <= 120)
    pseudo_img[mask_2] = [30, 144, 255]

    mask_3 = (gray_array >= 121) & (gray_array <= 180)
    pseudo_img[mask_3] = [50, 205, 50]

    mask_4 = (gray_array >= 181) & (gray_array <= 255)
    pseudo_img[mask_4] = [220, 20, 60]

    # Replicamos o gray_array para 3 canais apenas para facilitar a exibição lado a lado na UI
    gray_img_3d = np.dstack((gray_array, gray_array, gray_array))

    return gray_img_3d, pseudo_img

def color_redistribution(img_array: np.ndarray) -> np.ndarray:
    """ Falsa cor: Mapeia Verde->R, Azul->G, Vermelho->B """
    R_orig = img_array[:, :, 0]
    G_orig = img_array[:, :, 1]
    B_orig = img_array[:, :, 2]
    
    img_falsa_cor = np.dstack((G_orig, B_orig, R_orig))
    
    return img_falsa_cor