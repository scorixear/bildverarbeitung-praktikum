import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img = plt.imread("Woche_6/skeleton.png")
    # convert to grayscale if necessary
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # if values of pixels are 0-1, convert to 0-255
    if img.max() <= 1:
        img = np.mat(img) * 255
    img = img.astype('uint8')

    # apply binary filter
    binary = np.where(img > 127, 1, 0).astype('uint8')
    oldPic = binary.copy()
    # and count the zeros
    prevZeros = cv.countNonZero(oldPic)
    # apply first thinning
    thinned = thin(oldPic)
    zeros = cv.countNonZero(thinned)
    
    # while thinning removed more pixels
    # and we didn't remove all pixels
    while zeros != 0 and zeros != prevZeros:
        print("Recalculating thinning",zeros)
        # oldPic becomes the previously thinned
        oldPic = thinned
        # thin again
        thinned = thin(oldPic)
        # and count the zeros
        prevZeros = zeros
        zeros = cv.countNonZero(thinned)
    
    plt.subplot(121)
    plt.imshow(binary, cmap="gray")
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(oldPic, cmap="gray")
    plt.title("Thinned")
    plt.show()

def thin(img):
    # thinning is achieved by performing 6 iterations of the following:
    # Img = Img AND NOT K_x
    # where Kx is the image convolved with the corresponding kernels in a hitmiss operation
    newimg = cv.bitwise_and(img,    cv.bitwise_not(cv_kernel(img, np.array([[0,None,1],[0,1,1],[0,None,1]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,None],[0,1,1],[None,1,1]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[0,0,0],[None,1,None],[1,1,1]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,0,0],[1,1,0],[1,1,None]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,None,0],[1,1,0],[1,None,0]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,None],[1,1,0],[None,0,0]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[1,1,1],[None,1,None],[0,0,0]]))))
    newimg = cv.bitwise_and(newimg, cv.bitwise_not(cv_kernel(img, np.array([[None,1,1],[0,1,1],[0,0,None]]))))
    return newimg

def cv_kernel(img, matrix):
    # convert NONE to 0 and 0 to -1, as this is how cv2 interprets the kernel
    matrix[matrix == 0] = -1
    matrix[matrix == None] = 0
    # and apply the hitmiss kernel
    return cv.morphologyEx(img, cv.MORPH_HITMISS, matrix.astype(int))


if __name__ == "__main__":
    main()
