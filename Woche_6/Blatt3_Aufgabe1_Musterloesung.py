import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.signal as sig
import scipy.special as spec

def main():
    img = plt.imread("Woche_6/skeleton.png")
    # convert to grayscale
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # resize image to 50% and 200%
    img_50 = cv.resize(img, ((int(img.shape[1]*0.5)), int(img.shape[0]*0.5)), interpolation=cv.INTER_AREA)
    img_200 = cv.resize(img, ((int(img.shape[1]*2)), int(img.shape[0]*2)), interpolation=cv.INTER_AREA)
    
    # apply filters
    show_images(img)
    show_images(img_50)
    show_images(img_200)
    
    ## e) preferences
    # laplace is good in detecting high-intensity variations, but it is very sensitive to noise
    # sobel produces cleaner edges than laplace, but may not detect edges that are diagonal
    # LoG is good at detecting edges of various scales, but might miss edges that are parallel to the kernel axis
    # DoG follows LoGs path, but might produce thicker adges than necessary
    
    # computationally laplace and sobel are the cheapest, while dog and log are more expensive
    
def show_images(img):
    plt.subplot(231)
    plt.imshow(img, cmap="gray")
    plt.title('Original')
    
    plt.subplot(232)
    plt.imshow(apply_kernel(img, laplace()), cmap="gray")
    plt.title('Laplace')
    
    plt.subplot(233)
    plt.imshow(sobel(img), cmap="gray")
    plt.title('Sobel')
    
    plt.subplot(234)
    plt.imshow(apply_kernel(img, loG()), cmap="gray")
    plt.title('LoG')
    
    plt.subplot(235)
    plt.imshow(apply_kernel(img, doG()), cmap="gray")
    plt.title('DoG')
    
    plt.show()

def apply_kernel(img, kernel):
    return cv.filter2D(img, -1, kernel)

def laplace():
    d2x = np.array([[1,-2,1]])
    d2y = np.reshape(d2x, (3,1))

    d2x_pad = np.pad(d2x, ((1,1),(0,0)))
    d2y_pad = np.pad(d2y, ((0,0),(1,1)))
    return d2x_pad+d2y_pad

def sobel(img):
    gx = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    gy = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    imggx = cv.filter2D(img, -1, gx)
    imggy = cv.filter2D(img, -1, gy)
    
    return np.clip(np.sqrt(np.square(imggx) + np.square(imggy)),0,255)

def binomial(size: int):
    # generate a 2d array of ones
    gauss = np.ones((size, size))
    # for each cell in the matrix
    for i in range(size):
        for j in range(size):
            # apply the binomial formula: 1/(2^(2m))(m over i)(m over j)
            # where m is the size of the matrix - 1
            # and i,j are the current cell coordinates
            gauss[i,j] = (spec.binom(size-1, i)*spec.binom(size-1, j))
    return gauss * (1/(2**(2*(size-1))))

def loG():
    laplacian = laplace()
    gaussian = binomial(3)
    return sig.convolve2d(laplacian, gaussian, mode='full')
    
def doG():
    gaussian = binomial(3)
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
    return 4*sig.convolve2d((gaussian-identity),gaussian, mode='full')

if __name__ == "__main__":
    main()
