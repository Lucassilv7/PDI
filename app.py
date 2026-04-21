from impl.C3 import color_decoupling, show_images

if __name__ == "__main__":
    img_src = "assets/car.png"
    r_img, g_img, b_img, recoupled_img = color_decoupling(img_src)
    show_images((r_img, g_img, b_img, recoupled_img), titles=("Red Channel", "Green Channel", "Blue Channel", "Recoupled Image"))