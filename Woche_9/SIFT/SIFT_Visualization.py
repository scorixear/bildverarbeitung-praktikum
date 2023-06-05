import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from SIFT_KeyPoint import SIFT_KeyPoint

def visualize_scale_space(array: list[list[NDArray[np.float32]]], title: str):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(num=title, figsize=(1200*px,800*px))
    for o, octave in enumerate(array):
        for s, scale in enumerate(octave):
            plt.subplot(len(array), len(array[0]), o * len(array[0]) + s + 1)
            plt.title(f"O {o} S {s}")
            plt.imshow(scale, cmap="gray")
    plt.show()
    
def visualize_keypoints(scale_space: list[list[NDArray[np.float32]]],
                  keypoints: list[SIFT_KeyPoint],
                  deltas: list[float],
                  title: str,
                  use_keypoint_coordinates: bool = False,
                  show_orientations: bool = False,
                  show_descriptors: bool = False):
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    plt.figure(num=title, figsize=(1200*px,800*px))
    
    for o, octave in enumerate(scale_space):
        for s, scale in enumerate(octave):
            plt.subplot(len(scale_space), len(scale_space[0]), o * len(scale_space[0]) + s + 1)
            plt.title(f"O {o} S {s}")
            plt.imshow(scale, cmap="gray")
            keypoint_x = []
            keypoint_y = []
            quiver_x = []
            quiver_y = []
            descriptors = []
            if use_keypoint_coordinates:
                for keypoint in keypoints:
                    if keypoint.o == o and keypoint.s == s:
                        keypoint_x.append(round(keypoint.x/deltas[keypoint.o]))
                        keypoint_y.append(round(keypoint.y/deltas[keypoint.o]))
                        quiver_x.append(np.cos(keypoint.theta))#*keypoint.magnitude)
                        quiver_y.append(np.sin(keypoint.theta))#*keypoint.magnitude)
                        descriptors.append(keypoint.descriptor)
            else:
                keypoint_x = [keypoint.m for keypoint in keypoints if keypoint.o == o and keypoint.s == s]
                keypoint_y = [keypoint.n for keypoint in keypoints if keypoint.o == o and keypoint.s == s]
                quiver_x = [np.cos(keypoint.theta)*keypoint.magnitude for keypoint in keypoints if keypoint.o == o and keypoint.s == s]
                quiver_y = [np.sin(keypoint.theta)*keypoint.magnitude for keypoint in keypoints if keypoint.o == o and keypoint.s == s]
                descriptors = [keypoint.descriptor for keypoint in keypoints if keypoint.o == o and keypoint.s == s]
            plt.scatter(keypoint_x, keypoint_y, c="r", s=1)
            if(show_orientations):
                plt.quiver(keypoint_x, keypoint_y, quiver_x, quiver_y)
            ## print descriptors
            if(show_descriptors):
                print(f"Octave: {o}, extremum: {s}")
                print(np.matrix(descriptors))
    plt.show()
