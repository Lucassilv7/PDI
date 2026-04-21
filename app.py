from impl.C3 import *

if __name__ == "__main__":
    img_src = "assets/car.png"

    # Exibe a janela com os canais YUV
    display_yuv_tinted(img_src)
    
    # Quando você fechar a janela do YUV, ele abrirá a janela do HSV
    display_hsv_tinted(img_src)