import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy

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
    plt.imshow(laplace(img), cmap="gray")
    plt.title('Laplace')
    
    plt.subplot(233)
    plt.imshow(sobel(img), cmap="gray")
    plt.title('Sobel')
    
    plt.subplot(234)
    plt.imshow(loG(img), cmap="gray")
    plt.title('LoG')
    
    plt.subplot(235)
    plt.imshow(doG(img), cmap="gray")
    plt.title('DoG')
    
    plt.show()

def apply_kernel(img, kernel):
    return cv.filter2D(img, -1, kernel)

def laplace(img):
    d2x = np.array([1,-2,1])
    d2y = np.reshape(d2x, (3,1))

    d2x_pad = np.pad(d2x, ((1,1),(0,0)))

    #kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return kernel

def sobel(img):
    gx = (1/8)*np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    gy = (1/8)*np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    imggx = cv.filter2D(img, -1, gx)
    imggy = cv.filter2D(img, -1, gy)
    return np.sqrt(np.square(imggx) + np.square(imggy))

def loG(img):
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    kernel = scipy.signal.convolve2d(laplacian, gaussian, mode='full')
    print(kernel)
    #kernel = (1/16)*np.array([[0,1,2,1,0],[1,0,-2,0,1],[2,-2,-8,-2,2],[1,0,-2,0,1],[0,1,2,1,0]])
    return cv.filter2D(img, -1, kernel)
    
def doG(img):
    kernel = np.array([[1,4,6,4,1],[4,0,-8,0,4],[6,-8,-28,-8,6],[4,0,-8,0,4],[1,4,6,4,1]])
    return cv.filter2D(img, -1, kernel)

if __name__ == "__main__":
    main()
