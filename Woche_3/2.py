import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.special 


def getGauss(size: int):
  gauss = np.ones((size+1, size+1))
  for i in range(size+1):
    for j in range(size+1):
      gauss[i,j] = (scipy.special.binom(size, i)*scipy.special.binom(size,j))
  return gauss * (1/(2**(2*size)))

def main():
  image = plt.imread("test2.jpg")
  gauss = getGauss(7)
  print(gauss)
  filtered = cv.filter2D(image, -1, gauss)
  plt.imshow(filtered)
  plt.show()
  
if __name__ == "__main__":
  sys.exit(main())