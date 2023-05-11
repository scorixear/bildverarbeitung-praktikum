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
import cv2 as cv
import scipy.special


def main():
    # ----------- 3. Gaussian Kernel -------------
    image = plt.imread("Woche_3/test2.jpg")

    # generate Gauss filter for all sizes
    gauss_3 = get_gauss(3, 1)
    print(gauss_3)
    gauss_7 = get_gauss(7, 2)
    print(gauss_7)
    gauss_15 = get_gauss(15, 3)
    print(gauss_15)

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

def get_binomial(size: int):
    """Generates a Binomial Filter of the given size
    this approximates a gaussian filter

    Args:
        size (int): The Size of that filter

    Returns:
        npt.NDArray[np.float64]: The generated filter
    """
    # generate a 2d array of ones
    gauss = np.ones((size, size))
    # for each cell in the matrix
    for i in range(size):
        for j in range(size):
            # apply the binomial formula: 1/(2^(2m))(m over i)(m over j)
            # where m is the size of the matrix - 1
            # and i,j are the current cell coordinates
            gauss[i,j] = (scipy.special.binom(size-1, i)*scipy.special.binom(size-1, j))
    return gauss * (1/(2**(2*(size-1))))

def get_gauss(size: int, sigma: float):
    """Generates a Gaussian Filter of the given size

    Args:
        size (int): The Size of that filter
        sigma (float): The sigma value of the gaussian
    
    Returns:
        npt.NDArray[np.float64]: The generated filter
    """
    gauss = np.ones((size, size))
    half = size//2
    for i in range(-half, half+1, 1):
        for j in range(-half, half+1, 1):
            gauss[i+half,j+half] = (1/(2*np.pi*(sigma**2)))*np.exp(-1*((i**2+j**2)/(2*sigma**2)))
    return gauss / np.sum(gauss)

def show_images(images: list[tuple]):
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

def to_gray(image):
    """Converts an image to grayscale

    Args:
        image (cv.Mat): The color image

    Returns:
        cv.Mat: The grayscale image
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
    main()
    