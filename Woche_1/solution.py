# Imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    # 1. Show Pictures
    image = Image.open("Woche_1/test.jpg")
    numpy_data = np.asarray(image, dtype=np.uint8)
    plt.axis("off")
    plt.imshow(numpy_data)
    plt.show()

    # 2. Tiling
    tiled_image = tile_image(numpy_data, 3, 4)
    plt.axis("off")
    plt.imshow(tiled_image)
    plt.show()

    # 3.
    cropped_image = crop(numpy_data, 100, 100, 200, 200)
    plt.axis("off")
    plt.imshow(cropped_image)
    plt.show()

def tile_image(image, rows, cols):
    if cols == 1:
        tiled_image = image
    else:
        lst_images = []
        for _ in range(cols):
            lst_images.append(image)
        tiled_image = np.concatenate(lst_images, axis=1)
    if rows > 1:
        lst_images = []
        for _ in range(rows):
            lst_images.append(tiled_image)
        tiled_image = np.concatenate(lst_images, axis=0)
    return tiled_image

def crop(image, x1, y1, x2, y2):
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

if __name__ == "__main__":
    main()
