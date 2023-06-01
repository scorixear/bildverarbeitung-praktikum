import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from scipy import signal

class Extrema:
    def __init__(self, 
                 m: int,
                 n: int,
                 o: int,
                 s: int,
                 x: int = 0,
                 y: int = 0,
                 sigma: float = 0.0,
                 w: float = 0.0,
                 orientation: float = 0.0,
                 magnitude: float = 0.0,
                 descriptor: np.ndarray = np.array([])):
        self.m = m
        self.n = n
        self.o = o
        self.s = s
        self.x = x
        self.y = y
        self.sigma = sigma
        self.w: float = w
        self.orientation  = orientation
        self.magnitude = magnitude
        self.descriptor = descriptor
    def __str__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, d: {self.sigma}, x: {self.x}, y: {self.y}, dog: {self.w}, ori: {self.orientation}, mag: {self.magnitude}]"

    def __repr__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, d: {self.sigma}, x: {self.x}, y: {self.y}, dog: {self.w}, ori: {self.orientation}, mag: {self.magnitude}]"

def create_scale_space(img, octaves, scales):
    """Creates a scale space for a given image

    Args:
        img (np.ndarray): the image
        octaves (int): the number of octaves
        scales (int): the number of scales

    Returns:
        np.ndarray: the scale space
    """
    # create the scale space
    scale_space = []
    delta_min = 0.5
    delta_current = delta_min
    sigma_min = 0.8
    # scale image to double size
    resized_img = cv.resize(img, (0,0), fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    # last_scale_image respresents image at current scale level
    # calculate first gaussian blur here
    last_scale_image = cv.GaussianBlur(resized_img, (0,0), sigma_min)
    # previous_octave_image represents antepenultimate scale level of previous octave
    previous_octave_image = last_scale_image
    # for each octave
    for octave in range(octaves):
        octave_images = []
        # for each scale level
        for scale in range(scales):
            # sigma = 2**(octave + (scale / num_scales))
            # apply the kernel
            if(scale != 0 or octave != 0):
                sigma =(delta_current/delta_min)*sigma_min*2**((scale / (scales-2)))
                #ksize = (int(2*round(3*sigma)+1), int(2*round(3*sigma)+1))
                last_scale_image = cv.GaussianBlur(last_scale_image, (0,0), sigma)
            # add the filtered image to the scale space
            # if we are at the second last scale level of the octave
            octave_images.append(last_scale_image)
            if(scale == scales - 3):
                # save image for next octave
                previous_octave_image = cv.resize(last_scale_image, (0,0), fx=0.5, fy=0.5) # type: ignore
        last_scale_image = previous_octave_image
        scale_space.append(octave_images)
        delta_current = delta_min*2**(octave)
    # return the scale space
    return scale_space

def create_dogs(scale_space):
    """Creates the difference of gaussians for a given scale space

    Args:
        scale_space (np.ndarray): the scale space

    Returns:
        np.ndarray: the difference of gaussians
    """
    dogs = []
    # for each octave
    for octave in scale_space:
        dog_row = []
        # for each scale level (excluding last image)
        for i in range(len(octave)-1):
            # calculate difference of gaussians
            dog_row.append(np.subtract(octave[i+1], octave[i]))
        # results ins MAX_SCALE - 1 images per octave
        dogs.append(dog_row)
    return dogs

def find_discrete_extremas(dogs: list[list[np.ndarray]]) -> list[Extrema]:
    """Finds the discrete extremas for a given difference of gaussians

    Args:
        dogs (list[list[np.ndarray]]): the difference of gaussians

    Returns:
        list[Extrema]: the discrete extremas
    """
    extremas = []
    # for each octave in dogs
    for (octave_index, dog_octave) in enumerate(dogs):
        print(f"Octave {octave_index}")
        # for each dog image in the octave excluding first and last image
        for scale_index in range(1, len(dog_octave)-1):
            current = dog_octave[scale_index]
            before = dog_octave[scale_index-1]
            after = dog_octave[scale_index+1]
            for n in range(1, current.shape[0]-1):
                for m in range(1, current.shape[1]-1):
                    neighbours = current[n-1:n+2, m-1:m+2].flatten()
                    neighbours = np.delete(neighbours, 4)
                    neighbours = np.append(neighbours, before[n-1:n+2, m-1:m+2].flatten())
                    neighbours = np.append(neighbours, after[n-1:n+2, m-1:m+2].flatten())
                    if(np.max(neighbours) < current[n,m] or np.min(neighbours) > current[n,m]):
                        extremas.append(Extrema(m,n,octave_index,scale_index+1))
    # results in MAX_SCALE-3 possible extrema images
    return extremas

def taylor_expansion(extremas: list[Extrema], dog_scales: list[list], drop_off: float, contrast_threshold: float) -> list[Extrema]:
    new_extremas = []
    for extrema in extremas:
        # discard low contrast candiate keypoints
        if(np.abs(dog_scales[extrema.o][extrema.s-1][extrema.n, extrema.m]) < 0.8 * contrast_threshold):
            continue
        cs = extrema.s
        cm = extrema.m
        cn = extrema.n
        for _ in range(5):
            current = dog_scales[extrema.o][cs-1]
            previous = dog_scales[extrema.o][cs-2]
            next_scale = dog_scales[extrema.o][cs]
            
            first_derivative = np.matrix([[current[cn, cm+1] - current[cn, cm-1], current[cn+1, cm]- current[cn-1, cm], next_scale[cn, cm] - previous[cn, cm]]]).T
            
            dxx = (current[cn, cm+1] + current[cn, cm-1] - 2*current[cn, cm])
            dyy = (current[cn+1, cm]- current[cn-1, cm] - 2*current[cn, cm])
            dss = (next_scale[cn, cm] - previous[cn, cm] - 2*current[cn, cm])
            
            dxy = (current[cn+1, cm+1] - current[cn+1, cm-1]-current[cn-1, cm+1] - current[cn-1, cm-1])/4
            dxs = (next_scale[cn, cm+1] - next_scale[cn, cm-1]- previous[cn, cm+1] - previous[cn, cm-1])/4
            dys = (next_scale[cn+1, cm] - next_scale[cn-1, cm]- previous[cn+1, cm] - previous[cn-1, cm])/4
            
            hessian = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
            if(np.linalg.det(hessian) == 0):
                break
            alpha = -np.linalg.inv(hessian) @ first_derivative
            # accept extrema
            if(np.max(np.abs(alpha)) <= drop_off):
                w = current[extrema.n, extrema.m] + 0.5*alpha.T*first_derivative
                delta_min = 0.5
                delta_current = delta_min*2**(extrema.o)
                sigma_min = 0.8
                sigma = delta_current/delta_min*sigma_min*2**((alpha[2,0]+cs)/MAX_SCALE-2)
                x = delta_current*(alpha[0,0]+cm)
                y = delta_current*(alpha[1,0]+cn)
                
                new_extrema = Extrema(cm, cn, extrema.o, cs, x, y, sigma, w)
                new_extremas.append(new_extrema)
                break
            # reject extrema if offset is beyond image border
            if(round(cm+alpha[0,0]) < 1
               or round(cm+alpha[0,0]) >= current.shape[1]-1):
                break
            if(round(cn+alpha[1,0]) < 1
               or round(cn+alpha[1,0]) >= current.shape[0]-1):
                break
            if(round(cs+alpha[2,0]) < 1
               or round(cs+alpha[2,0]) >= len(dog_scales[extrema.o])-1):
                break
            cm = round(cm + alpha[0,0])
            cn = round(cn + alpha[1,0])
            cs = round(cs + alpha[2,0])
    return new_extremas

def filter_extremas(extremas: list[Extrema], dogs: list[list], contrast_threshold: float, curvature_threshold: float) -> list[Extrema]:
    filtered_extremas = []
    for extrema in extremas:
        cs = extrema.s
        current = dogs[extrema.o][cs-1]
        cx = extrema.m
        cy = extrema.n
        
        # contrast drop off
        if(abs(extrema.w) < contrast_threshold):
            continue
        # filter off edge extremas
        if(cx < 1 or cx > current.shape[1]-2 or cy < 1 or cy > current.shape[0]-2):
            continue
        
        dxx = (current[cy, cx+1] - current[cy, cx-1]) / 2
        dyy = (current[cy+1, cx]- current[cy-1, cx]) / 2
        
        dxy = (current[cy+1, cx+1] - current[cy+1, cx-1]-current[cy-1, cx+1] - current[cy-1, cx-1])/4
        H = np.matrix([[dxx, dxy], [dxy, dyy]])
        
        trace = np.trace(H)
        determinant = np.linalg.det(H)
        if(determinant == 0):
            continue
        curvature_ratio = (trace*trace)/determinant
        
        # curvature drop off
        if(curvature_ratio >= (curvature_threshold+1)**2/curvature_threshold):
            continue
        
        filtered_extremas.append(extrema)
    return filtered_extremas

def assign_orientations(extremas: list[Extrema], scale_space: list[list], window_size: int, num_bins: int) -> list[Extrema]:
    # create gaussian kernels for each scale
    new_extremas = []
    weight = np.exp(-(np.arange(window_size) - window_size // 2)**2 / (2 * 1.5**2))
    for extrema in extremas:
        # filter out extremas too close to edge
        # where window size does not fit in image
        if(extrema.m-window_size-1 < 0 
           or extrema.m+window_size+1 >= scale_space[extrema.o][extrema.s].shape[1]
           or extrema.n-window_size-1 < 0
           or extrema.n+window_size+1 >= scale_space[extrema.o][extrema.s].shape[0]):
            continue
        
        x = extrema.m
        y = extrema.n
        scale = extrema.s
        octave = extrema.o
        image = scale_space[octave][scale]
        
        xmin = x-window_size
        xmax = x+window_size
        ymin = y-window_size
        ymax = y+window_size
        
        local_region = image[ymin:ymax+1, xmin:xmax+1]
        
        dx = np.gradient(local_region, axis=1)
        dy = np.gradient(local_region, axis=0)
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx)
        
        histogram = np.zeros(num_bins)
        hist_width = np.pi*2 / num_bins
        
        for mag, angle, w in zip(magnitude.flat, orientation.flat, weight.flat):
            bin_index = round(angle/hist_width)%num_bins
            histogram[bin_index] += mag * w
        for _ in range(6):
            histogram = signal.convolve(histogram, np.array([1,1,1])/3)
        max_value = np.max(histogram)*0.8
        for index, h in enumerate(histogram):
            prev_hist = histogram[(index-1)%num_bins]
            next_hist = histogram[(index+1)%num_bins]
            if(h > prev_hist and h > next_hist and h > max_value):
                delta_k = 2*np.pi*(index)/num_bins
                delta_ref = delta_k + np.pi/num_bins*((prev_hist - next_hist)/(prev_hist + next_hist - 2*h))
                new_extrema = Extrema(extrema.m, extrema.n, extrema.o, extrema.s)
                new_extrema.w = extrema.w
                new_extrema.orientation = delta_ref
                new_extrema.magnitude = h
                new_extremas.append(new_extrema)
    return new_extremas

def create_descriptors(extremas: list[Extrema], scale_space: list[list], window_size, sub_window_size, num_bins, gradient_threshold) -> list[Extrema]:
    new_extremas = []
    gradient: list[Tuple[np.ndarray, np.ndarray]] = []
    for o in range(len(scale_space)):
        for s in range(len(scale_space[o])):
            image = scale_space[o][s]
            dx = np.gradient(image, axis=1)
            dy = np.gradient(image, axis=0)
            gradient.append((dx, dy))
    lambda_descr = window_size
    for extrema in extremas:
        limit = np.sqrt(2)*lambda_descr*extrema.s
        image = scale_space[extrema.o][extrema.s]
        if(round(limit*(sub_window_size+1)/sub_window_size) <= extrema.m 
           and extrema.m <= image.shape[1]-round(limit*(sub_window_size+1)/sub_window_size) 
           and round(limit*(sub_window_size+1)/sub_window_size) <= extrema.n 
           and extrema.n <= image.shape[0]-round(limit*(sub_window_size+1)/sub_window_size)):
            histograms = [[np.zeros(num_bins) for _ in range(sub_window_size)] for _ in range(sub_window_size)]
            for m in range(extrema.m - round(limit*(sub_window_size+1)/sub_window_size), extrema.m + round(limit*(sub_window_size+1)/sub_window_size), round(2*limit/sub_window_size)):
                for n in range(extrema.n - round(limit*(sub_window_size+1)/sub_window_size), extrema.n + round(limit*(sub_window_size+1)/sub_window_size), round(2*limit/sub_window_size)):
                    x_d = ((m-extrema.m)*np.cos(extrema.orientation) + (n-extrema.n)*np.sin(extrema.orientation)) / extrema.s
                    y_d = (-(m-extrema.m)*np.sin(extrema.orientation) + (n-extrema.n)*np.cos(extrema.orientation)) / extrema.s
                    
                    if(np.max(np.abs([x_d, y_d])) < lambda_descr*(sub_window_size+1)/sub_window_size):
                        grad = gradient[extrema.o * len(scale_space) + extrema.s]
                        delta_mn = (np.arctan2(grad[0][n,m], grad[1][n,m]) - extrema.orientation ) % (2*np.pi)
                        c_mn = np.exp(-(x_d**2 + y_d**2)/(2*(lambda_descr/sub_window_size)**2))
                        for i in range(sub_window_size):
                            if(np.abs(i-x_d) > (2*lambda_descr/sub_window_size)):
                                continue
                            for j in range(sub_window_size):
                                if(np.abs(j-y_d) > (2*lambda_descr/sub_window_size)):
                                    continue
                                for k in range(num_bins):
                                    if((2*np.pi*(k)/num_bins - delta_mn)%(2*np.pi/num_bins) >= (2*np.pi/num_bins)):
                                        continue
                                    temp = (1-(sub_window_size/(2*lambda_descr))*np.abs(x_d-i))
                                    temp *= (1-(sub_window_size/(2*lambda_descr))*np.abs(y_d-j))
                                    temp *= (1-(num_bins/(2*np.pi))*np.abs((delta_mn - (2*np.pi*(k)/num_bins))%(2*np.pi)))
                                    temp *= c_mn
                                    
                                    histograms[i][j][k] += temp
            feature_temp = []
            for i in range(sub_window_size):
                for j in range(sub_window_size):
                    for k in range(num_bins):
                        feature_temp.append(histograms[i][j][k])
            feature_vector = np.array(feature_temp)
            normalized = np.linalg.norm(feature_vector)
            for i in range(len(feature_vector)):
                feature_vector[i] = min(feature_vector[i], gradient_threshold*normalized)
                feature_vector[i] = min(np.floor(512*feature_vector[i]/normalized), 255)
            extrema.descriptor = feature_vector
            new_extremas.append(extrema)
    return new_extremas

EXTREMAS = []
SCALES = []
SCALE_OFFSET = 0
SCALE_BEGIN = 0
SCALE_END = 0
SHOW_ORIENTATIONS = False
SHOW_DESCRIPTORS = False
MAX_OCTAVE = 4
MAX_SCALE = 5
CURRENT_OCTAVE = 0
CURRENT_SCALE = 0

def show_array(array, title: str):
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
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # plot first image
    plt.title(f"Octave {CURRENT_OCTAVE} Scale {CURRENT_SCALE}")
    plt.imshow(array[0][0], cmap="gray")
    plt.show()
    fig.canvas.mpl_disconnect(cid)

def onclick(event):
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

def show_extremas(scale_space, 
                  extremas: list[Extrema], 
                  title: str, 
                  scale_offset: int = 1, 
                  scale_begin = 1, 
                  scale_end = MAX_SCALE-2, 
                  show_orientations: bool = False, 
                  show_descriptors: bool = False):
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    global SCALES
    global EXTREMAS
    global SCALE_OFFSET
    global SCALE_BEGIN
    global SCALE_END
    global SHOW_ORIENTATIONS
    global SHOW_DESCRIPTORS
    
    CURRENT_OCTAVE = 0
    CURRENT_SCALE = scale_begin
    SCALE_BEGIN = scale_begin
    SCALE_END = scale_end
    SCALES = scale_space
    EXTREMAS = extremas
    SCALE_OFFSET = scale_offset
    SHOW_ORIENTATIONS = show_orientations
    SHOW_DESCRIPTORS = show_descriptors
    
    fig = plt.figure(num=title)
    cid = fig.canvas.mpl_connect('button_press_event', show_extrema_onclick)
    # draw image
    plt.title(f"Octave {0} Scale {SCALE_BEGIN}")
    plt.imshow(SCALES[0][SCALE_BEGIN], cmap="gray")
    # draw extremas
    plt.scatter([extrema.m for extrema in EXTREMAS 
                    if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET], 
                [extrema.n for extrema in EXTREMAS 
                    if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET], c="r", s=1)
    if(SHOW_ORIENTATIONS):
        plt.quiver([extrema.m for extrema in EXTREMAS if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                   [extrema.n for extrema in EXTREMAS if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                   [np.sin(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                   [np.cos(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS if extrema.o == 0 and extrema.s == CURRENT_SCALE + SCALE_OFFSET])
    # if(SHOW_DESCRIPTORS):
    #     for extrema in EXTREMAS:
    #         if(extrema.octave == 0 and extrema.scale == CURRENT_SCALE + SCALE_OFFSET):
    #             for iy in range(4):
    #                 for ix in range(4):
    #                     descriptor = extrema.descriptors[iy*4+ix]
    #                     mag = 20#np.max(descriptor)
    #                     angle = np.argmax(descriptor)*360/8
    #                     plt.plot([extrema.x, 
    #                                extrema.x + np.sin(angle*np.pi/180)*mag], 
    #                                [extrema.y, 
    #                                extrema.y + np.cos(angle*np.pi/180)*mag],
    #                                color="y", linestyle="-", linewidth=0.5)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

def show_extrema_onclick(event):
    global CURRENT_OCTAVE
    global CURRENT_SCALE
    global SCALES
    global SCALE_OFFSET
    global EXTREMAS
    global SHOW_ORIENTATIONS
    
    CURRENT_SCALE = CURRENT_SCALE + 1
    if(CURRENT_SCALE == SCALE_END):
        CURRENT_OCTAVE = (CURRENT_OCTAVE + 1) % len(SCALES)
        CURRENT_SCALE = SCALE_BEGIN
    
    event.canvas.figure.clear()
    event.canvas.figure.gca().set_title(f"Octave {CURRENT_OCTAVE} Scale {CURRENT_SCALE}")
    event.canvas.figure.gca().imshow(SCALES[CURRENT_OCTAVE][CURRENT_SCALE], cmap="gray")
    event.canvas.figure.gca().scatter([extrema.m for extrema in EXTREMAS
                                       if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                      [extrema.n for extrema in EXTREMAS
                                       if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET], c="r", s=1)
    if(SHOW_ORIENTATIONS):
        event.canvas.figure.gca().quiver(   [extrema.m for extrema in EXTREMAS 
                                             if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [extrema.n for extrema in EXTREMAS 
                                             if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [np.sin(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS 
                                             if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [np.cos(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS 
                                             if extrema.o == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET])
    event.canvas.draw()
    
def detect_and_compute(img: np.ndarray, 
                       taylor_threshold: float = 0.6,
                       contrast_threshold: float = 0.015,
                       curvature_threshold: float = 10.0,
                       orientation_window_size: int = 10,
                       orientation_bins: int = 36,
                       descriptor_window_size: int = 16,
                       descriptor_sub_window_size: int = 4,
                       descriptor_bins: int = 8,
                       descriptor_gradient_threshold: float = 0.2,
                       show_plots: bool = False) -> Tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[Extrema]]:
    """detects and computes keypoints and descriptors for a given image

    Args:
        img (np.ndarray): the image to detect keypoints and descriptors for
        taylor_threshold (float, optional): describes maximum values for each axis
        of the offset vector from the taylor expansion. Defaults to 0.6.
        contrast_threshold (float, optional): extremas are rejected, if the difference of gaussian for that
        extremum or the local interpolated extremum is above. Defaults to 0.015.
        curvature_threshold (float, optional): extremas are rejected, if the curvature ratio (tr(H)**2/det(H))
        is above. Defaults to 10.0.
        orientation_window_size (int, optional): window size (one direction) for which the orientation is calculated
        With 10, this would result in 10 pixels in each direction considered. Defaults to 10.
        orientation_bins (int, optional): Number of Histogram bins for the orientation calculation. Should divide into 360. Defaults to 36.
        descriptor_window_size (int, optional): Area around each extrema, for which descriptors are calculated. Defaults to 16.
        descriptor_sub_window_size (int, optional): window size per descriptor. Should divide into descriptor_window_size. Defaults to 4.
        descriptor_bins (int, optional): Number of histogram bins for each descriptor. Should divide into 360. Defaults to 8.
        descriptor_gradient_threshold (float, optional): Histogram directions below this are set to 0 for each descriptor. Defaults to 0.2.
        show_plots (bool, optional): Shows Matplotlib plots for each step. Defaults to False.

    Returns:
        Tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[Extrema]]: scale space [octave][scale], dogs [octave, scale-1], extremas
    """
    scale_space = create_scale_space(img, MAX_OCTAVE, MAX_SCALE)
    dogs = create_dogs(scale_space)
    normalized_dogs = [[cv.normalize(d, None, 0, 1, cv.NORM_MINMAX) for d in octave] for octave in dogs] # type: ignore
    discrete_extremas = find_discrete_extremas(dogs)
    taylor_extremas = taylor_expansion(discrete_extremas, dogs, taylor_threshold, contrast_threshold)
    filtered_extremas = filter_extremas(taylor_extremas, dogs, contrast_threshold, curvature_threshold)
    key_points = assign_orientations(filtered_extremas, scale_space, orientation_window_size, orientation_bins)
    descriptor_key_points = create_descriptors(key_points, scale_space, descriptor_window_size, descriptor_sub_window_size, descriptor_bins, descriptor_gradient_threshold)
    
    if(show_plots):
        show_array(scale_space, "Scale Space")
        show_array(dogs, "Normalized DoGs")
        show_extremas(dogs, discrete_extremas, "Discrete Extremas")
        show_extremas(dogs, taylor_extremas, "Taylor Extremas")
        show_extremas(dogs, filtered_extremas, "Filtered Extremas")
        show_extremas(scale_space, key_points, "Key Points", 0, 2, MAX_SCALE-1, True)
        show_extremas(scale_space, descriptor_key_points, "Descriptor Key Points", 0, 2, MAX_SCALE-1, True, True)
    
    return scale_space, normalized_dogs, descriptor_key_points


def rotate_image(img: np.ndarray, angle: int):
    """rotates an image by a given angle

    Args:
        img (np.ndarray): the image to rotate
        angle (int): the angle to rotate by

    Returns:
        np.ndarray: the rotated image
    """
    rows, cols = img.shape
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    return cv.warpAffine(img, M, (cols, rows))

def main():
    # read in image
    img = cv.imread("Woche_9/cat.jpg")
    # resize image for faster processing
    img = cv.resize(img, (128, 104))
    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rotated = rotate_image(gray, 45)
    scale_space, normalized_dogs, key_points = detect_and_compute(gray, show_plots=True)
    rotated_scale_space, rotated_normalized_dogs, roated_key_points = detect_and_compute(rotated, show_plots=True)

    # Keypoint Matching
    
if __name__ == "__main__":
    main()
