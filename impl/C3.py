from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def show_images(images: tuple, titles: tuple = None) -> None:
    for i, img in enumerate(images):
        if titles:
            img.show(title=titles[i])
        else:
            img.show()

def color_decoupling(img_source: str) -> tuple:
    # O .convert("RGB") garante que a imagem terá 3 canais, 
    # evitando erros com PNGs que possuam canal Alpha (RGBA)
    img = Image.open(img_source).convert("RGB")
    img_array = np.array(img)

    # Cria matrizes zeradas com o mesmo shape (Altura, Largura, 3) e tipo (uint8) da original
    r_array = np.zeros_like(img_array)
    g_array = np.zeros_like(img_array)
    b_array = np.zeros_like(img_array)

    # Preenche apenas o canal de interesse (R=0, G=1, B=2), mantendo os outros em 0
    r_array[:, :, 0] = img_array[:, :, 0]  # R00
    g_array[:, :, 1] = img_array[:, :, 1]  # 0G0
    b_array[:, :, 2] = img_array[:, :, 2]  # 00B

    # A recomposição agora é a soma literal das matrizes
    recoupled_array = r_array + g_array + b_array

    # Converte as matrizes manipuladas de volta para objetos Image apenas para o retorno/exibição
    r_image = Image.fromarray(r_array)
    g_image = Image.fromarray(g_array)
    b_image = Image.fromarray(b_array)
    recoupled_image = Image.fromarray(recoupled_array)

    return r_image, g_image, b_image, recoupled_image

def rgb_to_cmyk(img_array: np.ndarray) -> tuple:
    # Normaliza a imagem para 0.0 - 1.0 para facilitar os cálculos
    rgb_norm = img_array.astype(np.float64) / 255.0

    # Extrai os canais R, G, B
    R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    # 1. RGB para CMY: Inverso aditivo
    C = 1.0 - R
    M = 1.0 - G
    Y = 1.0 - B

    # 2. CMY para CMYK
    # Encontra o K como o menor valor entre C, M e Y em cada pixel
    K = np.minimum(np.minimum(C, M), Y)

    # Evita divisão por zero criando uma máscara onde K é diferente de 1.0
    # Onde K for 1.0 (preto absoluto), a fórmula diz que CMYK = {0,0,0,1}
    mask = (K < 1.0)

    # Cria matrizes zeradas para C_new, M_new, Y_new
    C_k = np.zeros_like(C)
    M_k = np.zeros_like(M)
    Y_k = np.zeros_like(Y)

    # Aplica a fórmula apenas onde K < 1.0
    C_k[mask] = (C[mask] - K[mask]) / (1.0 - K[mask])
    M_k[mask] = (M[mask] - K[mask]) / (1.0 - K[mask])
    Y_k[mask] = (Y[mask] - K[mask]) / (1.0 - K[mask])

    return C_k, M_k, Y_k, K

def rgb_to_yuv(img_array: np.ndarray) -> tuple:
    # Pode manter em escala 0-255 aqui, já que são pesos lineares
    R = img_array[:, :, 0].astype(np.float64) / 255.0
    G = img_array[:, :, 1].astype(np.float64) / 255.0
    B = img_array[:, :, 2].astype(np.float64) / 255.0
    
    # Aplica os pesos para Y, U e V
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.312 * B
    
    # Retorna os três planos separados
    return Y, U, V

def rbg_to_hsv(img_array: np.ndarray) -> tuple:
    rgb_norm = img_array.astype(np.float64) / 255.0

    R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    # Encontra o valor máximo e mínimo de cor em cada pixel
    Cmax = np.max(rgb_norm, axis=2)
    Cmin = np.min(rgb_norm, axis=2)
    delta = Cmax - Cmin

    # 1. VALOR (V ou B)
    V = Cmax

    # 2. SATURAÇÃO (S)
    S = np.zeros_like(Cmax)
    # Se Cmax > 0, S = delta / Cmax. Senão, S = 0
    mask_s = Cmax > 0
    S[mask_s] = delta[mask_s] / Cmax[mask_s]

    # 3. MATIZ (H) - Calculado em graus (0 a 360)
    H = np.zeros_like(Cmax)
    mask_delta = delta > 0

    # Máscaras para saber qual canal é o máximo
    mask_R = mask_delta & (Cmax == R)
    mask_G = mask_delta & (Cmax == G)
    mask_B = mask_delta & (Cmax == B)
    
    # Aplica as fórmulas padronizadas do Hexacone
    H[mask_R] = 60.0 * (((G[mask_R] - B[mask_R]) / delta[mask_R]) % 6.0)
    H[mask_G] = 60.0 * (((B[mask_G] - R[mask_G]) / delta[mask_G]) + 2.0)
    H[mask_B] = 60.0 * (((R[mask_B] - G[mask_B]) / delta[mask_B]) + 4.0)

    return H, S, V

def display_cmyk_tinted(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    
    # Extrai os canais
    C, M, Y, K = rgb_to_cmyk(img_array)
    altura, largura = C.shape
    
    # --- TRUQUE DA COLORIZAÇÃO (Simulando tinta no monitor RGB) ---
    
    # 1. Ciano (Absorve o Red) -> Red = 1 - Ciano, Verde = 1, Azul = 1
    cyan_img = np.ones((altura, largura, 3)) # Começa como papel branco (tudo 1.0)
    cyan_img[:, :, 0] = 1.0 - C # Subtrai o vermelho onde tem ciano
    
    # 2. Magenta (Absorve o Green) -> Red = 1, Green = 1 - Magenta, Azul = 1
    magenta_img = np.ones((altura, largura, 3))
    magenta_img[:, :, 1] = 1.0 - M 
    
    # 3. Amarelo (Absorve o Blue) -> Red = 1, Green = 1, Blue = 1 - Amarelo
    yellow_img = np.ones((altura, largura, 3))
    yellow_img[:, :, 2] = 1.0 - Y
    
    # 4. Preto / Key (Absorve tudo) -> R, G e B recebem a mesma subtração
    black_img = np.ones((altura, largura, 3))
    black_img[:, :, 0] = 1.0 - K
    black_img[:, :, 1] = 1.0 - K
    black_img[:, :, 2] = 1.0 - K

    # --- PLOTANDO EXATAMENTE COMO NA SUA IMAGEM ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax in axes:
        ax.axis('off') # Tira as bordas
        
    axes[0].imshow(cyan_img)
    axes[0].set_title("Ciano (C)", fontsize=14)
    
    axes[1].imshow(magenta_img)
    axes[1].set_title("Magenta (M)", fontsize=14)
    
    axes[2].imshow(yellow_img)
    axes[2].set_title("Amarelo (Y)", fontsize=14)
    
    axes[3].imshow(black_img)
    axes[3].set_title("Preto (K)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def display_yuv_tinted(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    
    # Extrai Y, U, V usando a função base
    Y, U, V = rgb_to_yuv(img_array)
    
    # Função auxiliar para reverter de YUV para RGB para visualização no monitor
    def yuv_to_rgb(y, u, v):
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return np.clip(np.dstack((r, g, b)), 0, 1)

    # Cria as matrizes de apoio para colorização
    zero_uv = np.zeros_like(Y)
    mid_y = np.full_like(Y, 0.5) # Cinza médio para ver a diferença de cor

    # Colorização:
    # Y puro (tons de cinza)
    img_y = yuv_to_rgb(Y, zero_uv, zero_uv)
    # U puro (Luz neutra, V zerado)
    img_u = yuv_to_rgb(mid_y, U, zero_uv)
    # V puro (Luz neutra, U zerado)
    img_v = yuv_to_rgb(mid_y, zero_uv, V)

    # Plotagem
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax in axes: ax.axis('off')
    
    axes[0].imshow(img_array)
    axes[0].set_title("Original (RGB)", fontsize=14, fontweight='bold')
    
    axes[1].imshow(img_y)
    axes[1].set_title("Y (Luminância)", fontsize=14)
    
    axes[2].imshow(img_u)
    axes[2].set_title("U (Crominância B-Y)", fontsize=14)
    
    axes[3].imshow(img_v)
    axes[3].set_title("V (Crominância R-Y)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def display_hsv_tinted(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    
    H, S, V_hsv = rbg_to_hsv(img_array)
    
    ones = np.ones_like(H)
    H_norm = H / 360.0 
    
    # TRUQUE PARA O MATIZ: Usamos o V_hsv (Brilho original) em vez de "ones"!
    # Isso faz as sombras voltarem a ser escuras e esconde o ruído JPEG.
    img_h_limpo = hsv_to_rgb(np.dstack((H_norm, ones, V_hsv)))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax in axes: ax.axis('off')
    
    axes[0].imshow(img_array)
    axes[0].set_title("Original (RGB)", fontsize=14, fontweight='bold')
    
    # Matiz colorizado com as sombras originais
    axes[1].imshow(img_h_limpo)
    axes[1].set_title("H (Matiz Limpo)", fontsize=14)
    
    # Saturação em escala de cinza (Branco = Cor muito viva, Preto = Cinza/Desbotado)
    axes[2].imshow(S, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("S (Intensidade da Saturação)", fontsize=14)
    
    # Brilho em escala de cinza
    axes[3].imshow(V_hsv, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("V (Valor/Brilho)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def pseudocolor_density_slicing(img_path: str) -> None:
    # 1. Carrega a imagem original convertendo explicitamente para Tons de Cinza ('L')
    img_gray = Image.open(img_path).convert("L")
    gray_array = np.array(img_gray)

    # 2. Cria uma "tela em branco" RGB com as mesmas dimensões da imagem original
    altura, largura = gray_array.shape
    pseudo_img = np.zeros((altura, largura, 3), dtype=np.uint8)

    # 3. Definindo as Fatias (Condições) e as Cores

    # FATIA 1: Escuro (0 a 60) -> Cor: (160, 57, 0) [Laranja escuro/Marrom]
    mask_1 = (gray_array >= 0) & (gray_array <= 60)
    pseudo_img[mask_1] = [160, 57, 0]

    # FATIA 2: Tons médios baixos (61 a 120) -> Cor: (30, 144, 255) [Azul Dodger]
    mask_2 = (gray_array >= 61) & (gray_array <= 120)
    pseudo_img[mask_2] = [30, 144, 255]

    # FATIA 3: Tons médios altos (121 a 180) -> Cor: (50, 205, 50) [Verde Limão]
    mask_3 = (gray_array >= 121) & (gray_array <= 180)
    pseudo_img[mask_3] = [50, 205, 50]

    # FATIA 4: Claros (181 a 255) -> Cor: (220, 20, 60) [Vermelho Crimson]
    mask_4 = (gray_array >= 181) & (gray_array <= 255)
    pseudo_img[mask_4] = [220, 20, 60]

    # 4. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Exibe a imagem em tons de cinza original
    axes[0].imshow(gray_array, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Imagem Original (Tons de Cinza)", fontsize=14)
    axes[0].axis('off')
    
    # Exibe a imagem pseudocolorida
    axes[1].imshow(pseudo_img)
    axes[1].set_title("Pseudocolorização (Fatiamento por Densidade)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def color_redistribution(img_path: str):
    # 1. Carrega a imagem RGB
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    
    # 2. Extrai os canais originais
    R_orig = img_array[:, :, 0]
    G_orig = img_array[:, :, 1]
    B_orig = img_array[:, :, 2]
    
    # 3. A Redistribuição (Falsa Cor)
    
    # O np.dstack empilha os planos 2D na ordem que você passar (Novo R, Novo G, Novo B)
    img_falsa_cor = np.dstack((G_orig, B_orig, R_orig))
    
    # 4. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_array)
    axes[0].set_title("Original (Composição RGB Real)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_falsa_cor)
    axes[1].set_title("Redistribuição (Falsa Cor: G-B-R)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()