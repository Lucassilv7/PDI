import cv2
import numpy as np

def load_image(path: str):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem do caminho: {path}")
    
    return img

def aritmetica_imagens(img1, img2, operation):
    if img1.shape != img2.shape:
        raise ValueError("Imagens devem ter as mesmas dimensões para operações aritméticas")
    
    if operation == "add":
        result = cv2.add(img1, img2)
    elif operation == "subtract":
        result = cv2.subtract(img1, img2)
    elif operation == "multiply":
        result = cv2.multiply(img1, img2)
    elif operation == "divide":
        result = cv2.divide(img1, img2)
    else:
        raise ValueError("Operação não suportada. Use 'add', 'subtract', 'multiply', or 'divide'.")
    
    return result

def logica_imagens(img1, img2, operation):
    if img1.shape != img2.shape:
        raise ValueError("Imagem não tem as mesmas dimensões para operações lógicas")
    
    if operation == "and":
        result = cv2.bitwise_and(img1, img2)
    elif operation == "or":
        result = cv2.bitwise_or(img1, img2)
    elif operation == "xor":
        result = cv2.bitwise_xor(img1, img2)
    else:
        raise ValueError("Operação não suportada. Use 'and', 'or', ou 'xor'.")
    
    return result


if __name__ == "__main__":
    try:
        img = load_image(r"C:\Users\lucas\UFERSA\Computer_Science\PDI\Implementacao\python\imagens\Lenag.pgm")
        img2 = load_image(r"C:\Users\lucas\UFERSA\Computer_Science\PDI\Implementacao\python\imagens\aviao.png")

        img = cv2.resize(img, (512, 512))
        img2 = cv2.resize(img2, (512, 512))

        # Perform logical operation
        result = logica_imagens(img, img2, "xor")

        print("Imagem carregada com sucesso!")
        cv2.imshow("Imagem", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(e)