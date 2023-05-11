import numpy as np
from scipy.signal import convolve2d

L = np.pad(np.array([[1,-2,1]]), ((1,1),(0,0)))
B = np.array([[1],[-2],[1]])
print(np.add(L, B))



# Add padding of zeros to B to make it the same size as L
#convolution_matrix = convolve2d(L, B, mode='full')
#print(convolution_matrix)
