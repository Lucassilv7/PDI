from PIL import Image
import numpy as np

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