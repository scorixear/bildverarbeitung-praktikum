#------------------------------------------------#
#
# Blatt 2 - Musterlösung
# Aufgabe 3
# 20.04.2023
# Paul Keller
#
#------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import cv2 as cv
import scipy.special


def main():
    # ----------- 3. Gaussian Kernel -------------
    image = plt.imread("Woche_3/test2.jpg")

    # generate Gauss filter for all sizes
    gauss_3 = getGauss(3)
    gauss_7 = getGauss(7)
    gauss_15 = getGauss(15)

    # apply all filters to the image
    images = []
    filtered = cv.filter2D(image, -1, gauss_3)
    images.append((np.copy(image), np.copy(filtered)))
    filtered = cv.filter2D(image, -1, gauss_7)
    images.append((np.copy(image), np.copy(filtered)))
    filtered = cv.filter2D(image, -1, gauss_15)
    images.append((np.copy(image), np.copy(filtered)))

    # and show the images
    show_images(images)

def getGauss(size: int) -> npt.NDArray[np.float64]:
    """Generates a Gauss Filter of the given size

    Args:
        size (int): The Size of that filter

    Returns:
        npt.NDArray[np.float64]: The generated filter
    """
    # generate a 2d array of ones
    gauss = np.ones((size, size))
    print(gauss)
    # for each cell in the matrix
    for i in range(size):
        for j in range(size):
            # apply the gauss formula: 1/(2^(2m))(m over i)(m over j)
            # where m is the size of the matrix
            # and i,j are the current cell coordinates
            gauss[i,j] = (scipy.special.binom(size, i)*scipy.special.binom(size,j))
    return gauss * (1/(2**(2*size)))

def show_images(images: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]):
    """Shows A list of image tuples in a subplot

    Args:
        images (list[tuple[np.ndarray, np.ndarray]]): _description_
    """
    # This is really not necessary, you can also just repeat the code
    # iterate over each pair of images
    for i in range(len(images)):
        # show the images in one row
        plt.subplot(len(images), 2, i*2+1)
        plt.imshow(images[i][0])
        plt.subplot(len(images), 2, i*2+2)
        plt.imshow(images[i][1])
    plt.show()

def to_gray(image: cv.Mat) -> cv.Mat:
    """Converts an image to grayscale

    Args:
        image (cv.Mat): The color image

    Returns:
        cv.Mat: The grayscale image
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
    main()
    