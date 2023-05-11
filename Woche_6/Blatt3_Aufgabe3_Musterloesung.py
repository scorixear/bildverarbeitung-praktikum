import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img = plt.imread('Woche_6/cat2.jpg')
    # convert to grayscale if necessary
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # if values of pixels are 0-1, convert to 0-255
    if img.max() <= 1:
        img = np.mat(img) * 255
    img = img.astype('uint8')
    
    # apply binary filter
    binary = np.where(img > 127, 1, 0).astype('uint8')
    
    matrix_shape = (3,3)
    iterations = 1
    # erode the picture
    eroded = cv.erode(binary, np.ones(matrix_shape, np.uint8), iterations=iterations)
    # dilate the picture
    dilated = cv.dilate(binary, np.ones(matrix_shape, np.uint8), iterations=iterations)
    # opening is achieved by dilating the eroded picture
    opening = cv.dilate(eroded, np.ones(matrix_shape, np.uint8), iterations=iterations)
    # can be done by using cv method
    # opening_real = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones(matrix_shape, np.uint8), iterations=iterations)
    # closing is achieved by eroding the dilated picture
    closing = cv.erode(dilated, np.ones(matrix_shape, np.uint8), iterations=iterations)
    # can be done by using cv method
    # closing_real = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones(matrix_shape, np.uint8), iterations=iterations)

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.subplot(2, 3, 2)
    plt.imshow(binary, cmap="gray")
    plt.title("Binary")
    plt.subplot(2, 3, 3)
    plt.imshow(eroded, cmap="gray")
    plt.title("Eroded")
    plt.subplot(2, 3, 4)
    plt.imshow(dilated, cmap="gray")
    plt.title("Dilated")
    plt.subplot(2, 3, 5)
    plt.imshow(opening, cmap="gray")
    plt.title("Opening")
    plt.subplot(2, 3, 6)
    plt.imshow(closing, cmap="gray")
    plt.title("Closing")
    plt.show()


if __name__ == "__main__":
    main()
