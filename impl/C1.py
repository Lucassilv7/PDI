import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Carrega imagem como array NumPy RGB."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: str):
    """Salva array NumPy como imagem."""
    Image.fromarray(array.astype(np.uint8)).save(path)


def resize_to_match(img1: np.ndarray, img2: np.ndarray):
    """Redimensiona img2 para ter as mesmas dimensões de img1 (sem cv2)."""
    h, w = img1.shape[:2]
    pil = Image.fromarray(img2).resize((w, h), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


# =====================================================================
# OPERAÇÕES ARITMÉTICAS (manipulação de matriz pura)
# =====================================================================

def aritmetica_imagens(img1: np.ndarray, img2: np.ndarray, operation: str) -> np.ndarray:
    """
    Realiza operações aritméticas entre duas imagens usando NumPy puro.
    Parâmetros:
        img1, img2 : arrays uint8 de mesmas dimensões
        operation  : 'add' | 'subtract' | 'multiply' | 'divide'
    Retorna array uint8 com resultado saturado em [0, 255].
    """
    if img1.shape != img2.shape:
        img2 = resize_to_match(img1, img2)

    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        # Normaliza para manter resultado em [0,255]
        result = (a / 255.0) * (b / 255.0) * 255.0

    elif operation == "divide":
        # Evita divisão por zero
        safe_b = np.where(b == 0, 1, b)
        result = np.where(b == 0, 255.0, (a / safe_b) * 255.0)

    else:
        raise ValueError(f"Operação '{operation}' não suportada. Use 'add', 'subtract', 'multiply' ou 'divide'.")

    # Saturação: clamp em [0, 255]
    return np.clip(result, 0, 255).astype(np.uint8)


# =====================================================================
# OPERAÇÕES LÓGICAS (bit a bit com NumPy)
# =====================================================================

def logica_imagens(img1: np.ndarray, img2: np.ndarray, operation: str) -> np.ndarray:
    """
    Realiza operações lógicas bit a bit entre duas imagens.
    Parâmetros:
        img1, img2 : arrays uint8 de mesmas dimensões
        operation  : 'and' | 'or' | 'xor' | 'not'
    Retorna array uint8.
    """
    if operation != "not" and img1.shape != img2.shape:
        img2 = resize_to_match(img1, img2)

    if operation == "and":
        return np.bitwise_and(img1, img2)

    elif operation == "or":
        return np.bitwise_or(img1, img2)

    elif operation == "xor":
        return np.bitwise_xor(img1, img2)

    elif operation == "not":
        return np.bitwise_not(img1)

    else:
        raise ValueError(f"Operação '{operation}' não suportada. Use 'and', 'or', 'xor' ou 'not'.")
