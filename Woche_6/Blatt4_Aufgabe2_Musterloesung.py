import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# size of the pyramid
# first layer is source image
# followed by three resized images
PYRAMID_LEVELS = 4

def laplace(img):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    return cv.filter2D(img, -1, kernel)

def main():
    img = plt.imread("Woche_6/house.jpg")
    # convert to gray scale if necessary
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    pyramid = [img]
    # create a resized version for each pyramid level
    # by halfing the size of the previous image
    for i in range(PYRAMID_LEVELS-1):
        pyramid.append(cv.resize(pyramid[i], (pyramid[i].shape[1]//2, pyramid[i].shape[0]//2)))
    
    laplace_pyramid = []
    # create laplace images for each pyramid level
    for i in range(len(pyramid)):
        laplace_pyramid.append(laplace(pyramid[i]))
    
    reconstructedImages = []
    # reconstruct images
    for i in range(len(pyramid)-1, 0, -1):
        # by resizing the current level to the next higher level
        resize = cv.resize(pyramid[i], (laplace_pyramid[i-1].shape[1], laplace_pyramid[i-1].shape[0]))
        # retrieving the corresponding laplace image
        laplaceImage = laplace_pyramid[i-1]
        lMin = laplaceImage.min()
        lMax = laplaceImage.max()
        # map the laplace image from 0 to 255
        mappedLaplaceImage = np.interp(laplaceImage, [lMin, lMax], [0, 255])
        # and add it to the resized image
        reconstructed = (np.mat(resize) + mappedLaplaceImage)/2
        reconstructedImages.append((resize, laplaceImage, reconstructed))
    
    
    imageCounter = 1
    # show the created pyramid levels and their laplace images
    for i in range(len(pyramid)):
        plt.subplot(PYRAMID_LEVELS, 2,imageCounter)
        plt.title('Gaus '+str(i))
        plt.imshow(pyramid[i], cmap="gray")
        imageCounter+=1
        plt.subplot(PYRAMID_LEVELS, 2,imageCounter)
        plt.title('Laplace '+str(i))
        plt.imshow(laplace_pyramid[i], cmap="gray")
        imageCounter+=1
    plt.show()
    imageCounter=1
    # show the reconstructed images
    for i in range(len(reconstructedImages)):
        plt.subplot(PYRAMID_LEVELS-1, 3,imageCounter)
        plt.title('Resize Gaus '+str(i))
        plt.imshow(reconstructedImages[i][0], cmap="gray")
        imageCounter+=1
        plt.subplot(PYRAMID_LEVELS-1, 3,imageCounter)
        plt.title('Resize Laplace '+str(i))
        plt.imshow(reconstructedImages[i][1], cmap="gray")
        imageCounter+=1
        plt.subplot(PYRAMID_LEVELS-1, 3,imageCounter)
        plt.title('Reconstructed '+str(i))
        plt.imshow(reconstructedImages[i][2], cmap="gray")
        imageCounter+=1
    plt.show()
    imageCounter=1

if __name__ == "__main__":
    main()
