import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.special 


def main():
    # ----------- 1. Box Kernel -------------
    image = plt.imread('Woche_3/test2.jpg')
    kernel = np.ones((5,5))
    filtered = cv.filter2D(image, -1, kernel*1/25)
    show_images([(image, filtered)])
    images = []
    for center in range(2, 10):
        kernel[2,2] = center
        normalized = kernel * 1/(24 + center)
        filtered = cv.filter2D(image, -1, normalized)
        images.append((np.copy(image), np.copy(filtered)))
    show_images(images)

    # ----------- 2. Gaussian Kernel -------------
    image = plt.imread("Woche_3/test2.jpg")

    gauss_3 = getGauss(3)
    gauss_7 = getGauss(7)
    gauss_15 = getGauss(15)

    images = []
    filtered = cv.filter2D(image, -1, gauss_3)
    images.append((np.copy(image), np.copy(filtered)))
    filtered = cv.filter2D(image, -1, gauss_7)
    images.append((np.copy(image), np.copy(filtered)))
    filtered = cv.filter2D(image, -1, gauss_15)
    images.append((np.copy(image), np.copy(filtered)))

    show_images(images)

def getGauss(size: int):
    gauss = np.ones((size+1, size+1))
    for i in range(size+1):
        for j in range(size+1):
            gauss[i,j] = (scipy.special.binom(size, i)*scipy.special.binom(size,j))
    return gauss * (1/(2**(2*size)))

def show_images(images: list[tuple[np.ndarray, np.ndarray]]):
    for i in range(len(images)):
        plt.subplot(len(images), 2, i*2+1)
        plt.imshow(images[i][0])
        plt.subplot(len(images), 2, i*2+2)
        plt.imshow(images[i][1])
    plt.show()

# function for turning picture into gray scale
def to_gray(image):
  return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
  main()