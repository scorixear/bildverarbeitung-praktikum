import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys

# a) Read Image, 5x5 Box Filter, Plot
def a():
  an_image = plt.imread('C:/Users/pk/Dropbox/Master/MMI/2. Semester/Bildverarbeitung/Aufgaben/2/test2.jpg');
  filtered = cv.filter2D(an_image, -1, np.ones((5,5))*1/25)
  plt.subplot(1,2,1)
  plt.imshow(an_image)
  plt.subplot(1,2,2)
  plt.imshow(filtered)
  plt.show()


# b) 
def b(centerIncrease):
  an_image = plt.imread('C:/Users/pk/Dropbox/Master/MMI/2. Semester/Bildverarbeitung/Aufgaben/2/test2.jpg');
  boxfilter = np.ones((5,5));
  boxfilter[2,2] = boxfilter[2,2]+centerIncrease;
  totalMax = 1/(25+centerIncrease);
  filtered = cv.filter2D(an_image, -1, boxfilter*totalMax)
  plt.subplot(1,2,1)
  plt.imshow(an_image)
  plt.subplot(1,2,2)
  plt.imshow(filtered)
  plt.show()

def main():
  b(0)
  #a()

# function for turning picture into grey scale
def toGrey(image):
  return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
  sys.exit(main())