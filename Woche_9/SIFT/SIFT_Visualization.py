import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from SIFT_KeyPoint import KeyPoint
from SIFT_Params import SIFT_Params

EXTREMAS: list[KeyPoint] = []
SCALES: list[list[NDArray[np.float32]]] = []
SCALE_BEGIN: int = 0
SCALE_END: int = 0
SHOW_ORIENTATIONS: bool = False
SHOW_DESCRIPTORS: bool = False
CURRENT_OCTAVE: int = 0
CURRENT_SCALE: int = 0
USE_KEYPOINT_COORDINATES: bool = False
DELTAS: list[list[float]] = []
SIFT_PARAMS: SIFT_Params = SIFT_Params()

def visualize_scale_space(array: list[list[NDArray[np.float32]]], title: str):
    # introduce global variables
    global SCALES
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    # overwrite global variables
    SCALES = array
    CURRENT_OCTAVE = 0
    CURRENT_SCALE = 0
    # start figure
    fig = plt.figure(num=title)
    # and connect to onclick event
    cid = fig.canvas.mpl_connect('button_press_event', scale_space_on_click)
    # plot first image
    plt.title(f"Octave {CURRENT_OCTAVE} Scale {CURRENT_SCALE}")
    plt.imshow(array[0][0], cmap="gray")
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    
def scale_space_on_click(event):
    # introduce global variables
    global SCALES
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    
    # increment octave and scale
    CURRENT_SCALE = CURRENT_SCALE + 1
    if(CURRENT_SCALE == len(SCALES[CURRENT_OCTAVE])):
        CURRENT_OCTAVE = (CURRENT_OCTAVE + 1) % len(SCALES)
        CURRENT_SCALE = 0
    # redraw figure
    event.canvas.figure.clear()
    event.canvas.figure.gca().set_title(f"Octave {CURRENT_OCTAVE} Scale {CURRENT_SCALE}")
    event.canvas.figure.gca().imshow(SCALES[CURRENT_OCTAVE][CURRENT_SCALE], cmap="gray")
    event.canvas.draw()
    

def visualize_keypoints(scale_space: list[list[NDArray[np.float32]]],
                  extremas: list[KeyPoint],
                  title: str,
                  scale_end: int,
                  scale_begin: int = 1,
                  use_keypoint_coordinates: bool = False,
                  deltas: list[float] = [],
                  sift_params: SIFT_Params = SIFT_Params(),
                  show_orientations: bool = False, 
                  show_descriptors: bool = False):
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    global SCALES
    global EXTREMAS
    global SCALE_BEGIN
    global SCALE_END
    global SHOW_ORIENTATIONS
    global SHOW_DESCRIPTORS
    global USE_KEYPOINT_COORDINATES
    global DELTAS
    global SIFT_PARAMS
    
    CURRENT_OCTAVE = 0
    CURRENT_SCALE = scale_begin
    SCALE_BEGIN = scale_begin
    SCALE_END = scale_end
    SCALES = scale_space
    EXTREMAS = extremas
    SHOW_ORIENTATIONS = show_orientations
    SHOW_DESCRIPTORS = show_descriptors
    USE_KEYPOINT_COORDINATES = use_keypoint_coordinates
    DELTAS = deltas
    SIFT_PARAMS = sift_params
    
    fig = plt.figure(num=title)
    cid = fig.canvas.mpl_connect('button_press_event', keypoint_on_click)
    # draw image
    plt.title(f"Octave {0} Scale {SCALE_BEGIN}")
    plt.imshow(SCALES[0][SCALE_BEGIN], cmap="gray")
    # draw extremas
    
    keypoint_x = []
    keypoint_y = []
    quiver_x = []
    quiver_y = []
    descriptors = []
    if use_keypoint_coordinates:
        for keypoint in EXTREMAS:
            if keypoint.o == 0 and keypoint.s == CURRENT_SCALE:
                keypoint_x.append(round(keypoint.x/deltas[keypoint.o]))
                keypoint_y.append(round(keypoint.y/deltas[keypoint.o]))
                quiver_x.append(np.cos(keypoint.theta))#*keypoint.magnitude)
                quiver_y.append(np.sin(keypoint.theta))#*keypoint.magnitude)
                descriptors.append(keypoint.descriptor)
    else:
        keypoint_x = [keypoint.m for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
        keypoint_y = [keypoint.n for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
        quiver_x = [np.cos(keypoint.theta)*keypoint.magnitude for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
        quiver_y = [np.sin(keypoint.theta)*keypoint.magnitude for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
        descriptors = [keypoint.descriptor for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
    plt.scatter(keypoint_x, keypoint_y, c="r", s=1)
    # draw orientations
    if(SHOW_ORIENTATIONS):
        plt.quiver(keypoint_x, keypoint_y, quiver_x, quiver_y)
    ## print descriptors
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE}")
        print(np.matrix(descriptors))
    plt.show()
    fig.canvas.mpl_disconnect(cid)
def keypoint_on_click(event):
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    global SCALES
    global EXTREMAS
    global SHOW_ORIENTATIONS
    
    CURRENT_SCALE = CURRENT_SCALE + 1
    if(CURRENT_SCALE == SCALE_END):
        CURRENT_OCTAVE = (CURRENT_OCTAVE + 1) % len(SCALES)
        CURRENT_SCALE = SCALE_BEGIN
    
    event.canvas.figure.clear()
    event.canvas.figure.gca().set_title(f"Octave {CURRENT_OCTAVE} Scale {CURRENT_SCALE}")
    event.canvas.figure.gca().imshow(SCALES[CURRENT_OCTAVE][CURRENT_SCALE], cmap="gray")
    
    keypoint_x = []
    keypoint_y = []
    quiver_x = []
    quiver_y = []
    descriptors = []
    if USE_KEYPOINT_COORDINATES:
        for keypoint in EXTREMAS:
            if keypoint.o == CURRENT_OCTAVE and keypoint.s == CURRENT_SCALE:
                keypoint_x.append(round(keypoint.x/DELTAS[keypoint.o]))
                keypoint_y.append(round(keypoint.y/DELTAS[keypoint.o]))
                quiver_x.append(np.cos(keypoint.theta))#*keypoint.magnitude)
                quiver_y.append(np.sin(keypoint.theta))#*keypoint.magnitude)
                descriptors.append(keypoint.descriptor)
    else:
        keypoint_x = [keypoint.m for keypoint in EXTREMAS if keypoint.o == CURRENT_OCTAVE and keypoint.s == CURRENT_SCALE]
        keypoint_y = [keypoint.n for keypoint in EXTREMAS if keypoint.o == CURRENT_OCTAVE and keypoint.s == CURRENT_SCALE]
        quiver_x = [np.cos(keypoint.theta)*keypoint.magnitude for keypoint in EXTREMAS if keypoint.o == CURRENT_OCTAVE and keypoint.s == CURRENT_SCALE]
        quiver_y = [np.sin(keypoint.theta)*keypoint.magnitude for keypoint in EXTREMAS if keypoint.o == CURRENT_OCTAVE and keypoint.s == CURRENT_SCALE]
        descriptors = [keypoint.descriptor for keypoint in EXTREMAS if keypoint.o == 0 and keypoint.s == CURRENT_SCALE]
    event.canvas.figure.gca().scatter(keypoint_x, keypoint_y, c="r", s=1)
    if(SHOW_ORIENTATIONS):
        event.canvas.figure.gca().quiver(keypoint_x, keypoint_y, quiver_x, quiver_y)
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE}")
        print(np.matrix(descriptors))
    event.canvas.draw()