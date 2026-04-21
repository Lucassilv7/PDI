import numpy as np
import matplotlib.pyplot as plt
from impl.C3 import *

if __name__ == "__main__":
    img_array = image_to_array("assets/car.png")
    
    # Executa a matemática no Core
    h, s, v = get_hsv_clean_images(img_array)
    
    # Exibe no Matplotlib (ou no Streamlit com st.image)
    plt.imshow(h)
    plt.title("H Limpo vindo do Core")
    plt.show()