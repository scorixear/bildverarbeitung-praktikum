# Imports
import matplotlib.pyplot as plt # used for showing the pictures
import numpy as np # used for transforming the pictures
from PIL import Image # used for reading the pictures

def main():
    # 1. Show Pictures
    # files are always searched locally, meaning from the current working directory
    # The current working directory is always from where the python file is executed
    # = the folder you have open in VS Code
    # so in this case the python file is inside a folder called Woche_1
    image = Image.open("Woche_1/test.jpg")

    # converts an image to a numpy array, while casting every value
    # (color of each pixel) to a byte (uint8)
    numpy_data = np.asarray(image, dtype=np.uint8)
    # removes the axis from the plot
    plt.axis("off")
    # draws the numpy array to the plot
    plt.imshow(numpy_data)
    # shows the plot
    plt.show()

    # 2. Tiling
    # tiles the image 3 times in the y direction (rows) and 4 times in the x direction (cols)
    tiled_image = tile_image(numpy_data, 3, 4)
    plt.axis("off")
    plt.imshow(tiled_image)
    plt.show()

    # 3.
    # crops the image from (100, 100) to (200, 200)
    cropped_image = crop(numpy_data, 100, 100, 200, 200)
    plt.axis("off")
    plt.imshow(cropped_image)
    plt.show()


def tile_image(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tiles an image, by repeating it in the x and y direction

    Args:
        image (np.ndarray): the np array to tile
        rows (int): number of repetitions in the y direction
        cols (int): number of repetitions in the x direction

    Returns:
        np.ndarray: the tiled image
    """
    # if the number of columns is 1, we don't need to concatenate in axis=1 direction
    # since both if and else clause initialize tiled_image, we can use it later as the return value
    if cols == 1:
        tiled_image: np.ndarray = image
    # otherwise we concatenate the image cols times in the x direction
    else:
        lst_images = []
        for _ in range(cols):
            lst_images.append(image)
        tiled_image: np.ndarray = np.concatenate(lst_images, axis=1)
    # if the number of rows is bigger than 1, we concatenate the image cols times in the y direction
    if rows > 1:
        lst_images = []
        for _ in range(rows):
            lst_images.append(tiled_image)
        tiled_image: np.ndarray = np.concatenate(lst_images, axis=0)
    return tiled_image


def crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crops the image between two points

    Args:
        image (np.ndarray): the image to crop
        x1 (int): first point x coordinate
        y1 (int): first point y coordinate
        x2 (int): second point x coordinate
        y2 (int): second point y coordinate	

    Returns:
        np.ndarray: the cropped image
    """
    # the image is a 3d array, the np array indexing provides a
    # concise way to access every dimension [y, x, color]
    # we don't need to change the color dimension, so we leave it out
    # the ':' means we want to access all values between both values
    # [a:b] -> [a, a+1, a+2, ..., b-1]
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

# this ensures the main function is only run, if the python file is executed directly
# it also solves the problem with defining functions after the main function
# without this, you would always need to define a function before you call it
if __name__ == "__main__":
    main()
