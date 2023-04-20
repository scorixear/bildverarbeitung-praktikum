#------------------------------------------------#
#
# Blatt 2 - Musterlösung
# Aufgabe 2
# 20.04.2023
# Paul Keller
#
#------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import cv2 as cv


def main():
    # ----------- 2. Box Kernel -------------
    image = plt.imread('Woche_3/test2.jpg')
    # A 5x5 Matrix filled with 1 = BoxKernel
    kernel = np.ones((5,5))
    # Filter the image with the kernel, by scaling the kernel with 1/25
    filtered = cv.filter2D(image, -1, kernel*1/25)
    # and show the images
    show_images(image, filtered)
    # for center values from 2 to 100, with a step size of 10
    for center in range(1, 100, 10):
        # set the center value
        kernel[2,2] = center
        # normalize the kernel
        normalized = kernel * 1/(24 + center)
        # apply the filter
        filtered = cv.filter2D(image, -1, normalized)
        # and show the images
        show_images(image, filtered)

def show_images(image1: npt.NDArray[np.float64], image2: npt.NDArray[np.float64]):
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()

if __name__ == "__main__":
    main()