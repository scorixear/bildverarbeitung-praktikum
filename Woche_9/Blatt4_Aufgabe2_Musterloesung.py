import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
from scipy import signal

class Extremum:
    def __init__(self, 
                 m: int,
                 n: int,
                 o: int,
                 s: int,
                 x: int = 0,
                 y: int = 0,
                 sigma: float = 0.0,
                 omega: float = 0.0,
                 orientation: float = 0.0,
                 magnitude: float = 0.0,
                 descriptor: np.ndarray = np.array([])):
        """Represents Extrema / Keypoint

        Args:
            m (int): The x - coordinate in Dog image
            n (int): The y - coordinate in Dog image
            o (int): The octave
            s (int): The scale level from Dog image
            x (int, optional): The x - coordinate in scale space. Defaults to 0.
            y (int, optional): the y - coordinate in scale space. Defaults to 0.
            sigma (float, optional): The scale level in scale space. Defaults to 0.0.
            omega (float, optional): The scale value in scale space. Defaults to 0.0.
            orientation (float, optional): The Orientation delta in radians. Defaults to 0.0.
            magnitude (float, optional): The magnitude of the orientation. Defaults to 0.0.
            descriptor (np.ndarray, optional): The descriptor feature vector. Defaults to np.array([]).
        """
        self.m = m
        self.n = n
        self.o = o
        self.s = s
        self.x = x
        self.y = y
        self.sigma = sigma
        self.omega: float = omega
        self.orientation  = orientation
        self.magnitude = magnitude
        self.descriptor = descriptor
    def __str__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, d: {self.sigma}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, w: {self.omega}, ori: {self.orientation}, mag: {self.magnitude}]"

    def __repr__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, d: {self.sigma}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, w: {self.omega}, ori: {self.orientation}, mag: {self.magnitude}]"

def create_scale_space(img: np.ndarray, octaves: int, scales: int, delta_min: float = 0.5, sigma_min: float = 0.8) -> list[list[np.ndarray]]:
    """Creates a scale space for a given image

    Args:
        img (np.ndarray): the image
        octaves (int): the number of octaves
        scales (int): the number of scales
        delta_min (float): the starting delta value. Defaults to 0.5.
        sigma_min (float): the starting sigma value. Defaults to 0.8.

    Returns:
        list[list[np.ndarray]]: the scale space divide into octave - scales - images
    """
    # create the scale space
    scale_space = []
    # represent current scale factor
    delta_current = delta_min
    # scale image up with bilinear interpolation
    resized_img = cv.resize(img, (0,0), fx=1.0 / delta_current, fy=1.0 / delta_current, interpolation=cv.INTER_LINEAR)
    # last_scale_image respresents image at current scale level
    last_scale_image = resized_img
    # previous_octave_image represents antepenultimate scale level of previous octave
    previous_octave_image = None
    # for each octave
    for octave in range(octaves):
        octave_images = []
        # next scaling factor
        # will be delta_min for octave = 0
        delta_current = delta_min*2**(octave)
        # for each scale level
        for scale in range(scales):
            # calculate sigma
            # will be sigma_min for octave 0 and scale 0
            # will be (delta_current/delta_min)*sigma_min for scale = 0
            sigma =(delta_current/delta_min)*sigma_min*2**((scale / (scales-2)))
            # (0,0) ksize means calculate kernel size from sigma
            last_scale_image = cv.GaussianBlur(last_scale_image, (0,0), sigma)
            # add the filtered image to the scale space
            octave_images.append(last_scale_image)
            # if we are at the second last scale level of the octave
            if(scale == scales - 3):
                # calculate next scaling factor
                delta_next = delta_min*2**(octave+1)
                # sample down and
                # save image for next octave
                previous_octave_image = cv.resize(last_scale_image, (0,0), fx=delta_next / delta_current, fy=delta_next / delta_current) # type: ignore
        last_scale_image = previous_octave_image
        scale_space.append(octave_images)
    # return the scale space
    return scale_space

def create_dogs(scale_space: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    """Creates the difference of gaussians for a given scale space

    Args:
        scale_space (list[list[np.ndarray]]): the scale space

    Returns:
        list[list[np.ndarray]]: the difference of gaussians. Will result in scale-1 images per octave
    """
    dogs = []
    # for each octave
    for octave in scale_space:
        dog_row = []
        # for each scale level (excluding last image)
        for i in range(len(octave)-1):
            # calculate difference of gaussians
            # possibly use cv.subtract here?
            dog_row.append(np.subtract(octave[i+1], octave[i]))
        # results ins MAX_SCALE - 1 images per octave
        dogs.append(dog_row)
    return dogs

def find_discrete_extremas(dogs: list[list[np.ndarray]]) -> list[Extremum]:
    """Finds the discrete extremas for a given difference of gaussians

    Args:
        dogs (list[list[np.ndarray]]): the difference of gaussians

    Returns:
        list[Extrema]: the discrete extremas
    """
    extremas = []
    # for each octave in dogs
    for (octave_index, octave_images) in enumerate(dogs):
        print(f"Extrema Calculation: Octave {octave_index}")
        # for each dog image in the octave excluding first and last image
        for scale_index in range(1, len(octave_images)-1):
            # the current dog image
            current = octave_images[scale_index]
            # the one before
            before = octave_images[scale_index-1]
            # the one after
            after = octave_images[scale_index+1]
            # for each pixel in the image
            # exluding the border
            for n in range(1, current.shape[0]-1):
                for m in range(1, current.shape[1]-1):
                    # neighbours in current dog image
                    neighbours = current[n-1:n+2, m-1:m+2].flatten()
                    # excluding itself
                    neighbours = np.delete(neighbours, 4)
                    # neighbours in before dog image
                    neighbours = np.append(neighbours, before[n-1:n+2, m-1:m+2].flatten())
                    # neighbours in after dog image
                    neighbours = np.append(neighbours, after[n-1:n+2, m-1:m+2].flatten())
                    # if the current pixel is local minimum or maximum
                    if(np.max(neighbours) < current[n,m] or np.min(neighbours) > current[n,m]):
                        # create new local extremum
                        # scale index represent the index of the dog image
                        # For dog image i, we calculated scale image s_i+1 - s_i
                        # therefore, we take scale_index+1 here as the correct scale index
                        extremas.append(Extremum(m,n,octave_index,scale_index+1))
    # results in MAX_SCALE-3 possible extrema images per octave
    # since we cannot take 1st and last image of each octave
    return extremas

def taylor_expansion(extremas: list[Extremum], dog_scales: list[list[np.ndarray]], drop_off: float, contrast_threshold: float, delta_min: float = 0.5, sigma_min: float = 0.8) -> list[Extremum]:
    """Finetunes locations of extrema using taylor expansion

    Args:
        extremas (list[Extremum]): The extremas to finetune
        dog_scales (list[list[np.ndarray]]): The difference of gaussians images to finetune with
        drop_off (float): if offset is below, we accept a new location
        contrast_threshold (float): if dog value is below, we discard the extrema
        delta_min (float): the starting delta value. Defaults to 0.5.
        sigma_min (float): the starting sigma value. Defaults to 0.8.

    Returns:
        list[Extremum]: The new Extremum. Newly created Extremum Objects
    """
    new_extremas = []
    for extremum in extremas:
        # discard low contrast candiate keypoints
        if(np.abs(dog_scales[extremum.o][extremum.s-1][extremum.n, extremum.m]) < 0.8 * contrast_threshold):
            continue
        # location of the extremum
        # will be adjusted maximum 5 times
        # locations are 0-based
        # attention when calculating with values
        cs = extremum.s
        cm = extremum.m
        cn = extremum.n
        # for each adjustment
        # will break if new location is found
        for _ in range(5):
            # the dog image is always one lower than the scale index
            current = dog_scales[extremum.o][cs-1]
            previous = dog_scales[extremum.o][cs-2]
            next_scale = dog_scales[extremum.o][cs]
            # called $\bar{g}^o_{s,m,n}$ in the paper
            # represent the first derivative  in a finite difference scheme
            # is Transposed, as we calculate [a,b,c] values, but want [[a],[b],[c]]
            first_derivative = np.matrix([[next_scale[cn, cm] - previous[cn, cm], current[cn, cm+1] - current[cn, cm-1], current[cn+1, cm]- current[cn-1, cm]]]).T
            
            # calcuation of hessian matrix
            dxx = (current[cn, cm+1] + current[cn, cm-1] - 2*current[cn, cm])
            dyy = (current[cn+1, cm]- current[cn-1, cm] - 2*current[cn, cm])
            dss = (next_scale[cn, cm] - previous[cn, cm] - 2*current[cn, cm])
            
            # dxy, dxs and dys are reused for dyx, dsx and dsy
            # as they are the same value
            dxy = (current[cn+1, cm+1] - current[cn+1, cm-1]-current[cn-1, cm+1] - current[cn-1, cm-1])/4
            dxs = (next_scale[cn, cm+1] - next_scale[cn, cm-1]- previous[cn, cm+1] - previous[cn, cm-1])/4
            dys = (next_scale[cn+1, cm] - next_scale[cn-1, cm]- previous[cn+1, cm] - previous[cn-1, cm])/4
            
            # paper annotates 1 = s, 2 = x, 3 = y
            hessian = np.matrix([[dss, dxs, dxy], [dxs, dxx, dys], [dys, dxy, dyy]])
            # inverse of a matrix with det = 0 is not possible
            # therefore we break here
            if(np.linalg.det(hessian) == 0):
                break
            # calculate offset
            alpha = -np.linalg.inv(hessian) @ first_derivative
            # the every value is below the drop_off
            # we found the new location
            if(np.max(np.abs(alpha)) <= drop_off):
                # this is simplified from 'w+alphaT*g + 0.5*alphaT*H*alpha'
                # to 'w+0.5*alphaT*g' following the paper
                # pseudocode does not simplify here
                # omega represent the value of the DoG interpolated extremum
                omega = current[extremum.n, extremum.m] + 0.5*alpha.T*first_derivative
                # calculate the current delta and sigma for the corresponding new location
                delta_current = delta_min*2**(extremum.o)
                sigma = delta_current/delta_min*sigma_min*2**((alpha[0,0]+cs)/MAX_SCALE-2)
                # and the keypoint coordinates
                x = delta_current*(alpha[1,0]+cm)
                y = delta_current*(alpha[2,0]+cn)
                # create new Extremum object with the corresponding values
                new_extrema = Extremum(cm, cn, extremum.o, cs, x, y, sigma, omega)
                new_extremas.append(new_extrema)
                break
            # reject extrema if offset is beyond image border or scale border
            if(round(cm+alpha[0,0]) < 1
               or round(cm+alpha[0,0]) >= current.shape[1]-1):
                break
            if(round(cn+alpha[1,0]) < 1
               or round(cn+alpha[1,0]) >= current.shape[0]-1):
                break
            # since cs represents scale index, we need values from dog index, this is capped at [2, dog_scales]
            if(round(cs+alpha[2,0]) < 2
               or round(cs+alpha[2,0]) >= len(dog_scales[extremum.o])):
                break
            # if the new location is valid, update the locations in that direction
            # at least one value will be adjust (as at least one is >0.6)
            cs = round(cs + alpha[0,0])
            cm = round(cm + alpha[1,0])
            cn = round(cn + alpha[2,0])
    return new_extremas

def filter_extremas(extremas: list[Extremum], dogs: list[list], contrast_threshold: float, curvature_threshold: float) -> list[Extremum]:
    filtered_extremas = []
    for extrema in extremas:
        cs = extrema.s
        cx = extrema.m
        cy = extrema.n
        current = dogs[extrema.o][cs-1]
        
        # contrast drop off
        if(abs(extrema.omega) < contrast_threshold):
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

def scale_space_gradients(scale_space: list[list[np.ndarray]], get_m_limit: Callable[[int, int], int], get_n_limit: Callable[[int, int], int]) -> dict[Tuple[int, int, int, int], Tuple[float, float]]:
    gradient_2d: dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
    for o in range(len(scale_space)):
        for s in range(len(scale_space[o])-2):
            limit_m = get_m_limit(o, s)
            limit_n = get_n_limit(o, s)
            image = scale_space[o][s]
            for n in range(limit_m):
                for m in range(limit_n):
                    gradient_2d[o,s,n,m] = [((image[n,m+1]-image[n,m-1])/2),((image[n+1,m]-image[n-1,m])/2)]
    return gradient_2d

def assign_orientations(extremas: list[Extremum], scale_space: list[list[np.ndarray]], lambda_ori: float, num_bins: int) -> list[Extremum]:
    # create gaussian kernels for each scale
    new_extremas = []
    sigma_min = 0.8
    delta_min = 0.5
    m_limit: Callable[[int, int], int] = lambda o, s: round(3*lambda_ori*(delta_min*2**(o)/delta_min)*sigma_min*2**(s/(len(scale_space[o]-2))))
    gradient_2d: dict[Tuple[int, int, int, int], Tuple[float, float]] = scale_space_gradients(scale_space, m_limit, m_limit)
    
    for extrema in extremas:
        o = extrema.o
        x = extrema.x
        y = extrema.y
        delta_o = delta_min*2**(extrema.o)
        limit = 3*lambda_ori*extrema.sigma
        image = scale_space[extrema.o][round(extrema.sigma)]
        # filter out extremas too close to edge
        # where window size does not fit in image
        if(x-limit < 0 or x+limit >= image.shape[1] or y-limit < 0 or y+limit >= image.shape[0]):
            continue
        histogram = np.zeros(num_bins)
        for m in range(round((x-limit)/delta_o), round((x+limit)/delta_o)):
            for n in range(round((y-limit)/delta_o), round((y+limit)/delta_o)):
                dx = gradient_2d[o,round(extrema.sigma),n,m][0]
                dy = gradient_2d[o,round(extrema.sigma),n,m][1]
                c_ori = np.exp(-np.linalg.norm(np.array([m*delta_o, n*delta_o])-np.array([x,y]))**2/(2*(lambda_ori*delta_o)**2))*np.linalg.norm([dx,dy])
                bin_index = round((num_bins/(2*n.pi)*(np.arctan2(dx,dy)))%(2*np.pi))
                histogram[bin_index] += c_ori
        
        for _ in range(6):
            histogram = signal.convolve(histogram, np.array([1,1,1])/3)
        max_value = np.max(histogram)*0.8
        for k, h_k in enumerate(histogram):
            prev_hist = histogram[(k-1)%num_bins]
            next_hist = histogram[(k+1)%num_bins]
            if(h_k > prev_hist and h_k > next_hist and h_k > max_value):
                Delta_k = 2*np.pi*(k)/num_bins
                Delta_key = Delta_k+np.pi/num_bins*((prev_hist - next_hist)/(prev_hist + next_hist - 2*h_k))
                new_extrema = Extremum(extrema.m, extrema.n, extrema.o, extrema.s, x, y, extrema.sigma, extrema.omega, Delta_key, h_k)
                new_extremas.append(new_extrema)
    return new_extremas

def create_descriptors(extremas: list[Extremum], scale_space: list[list[np.ndarray]], lambda_descr: float, n_hist: int, n_ori: int, gradient_threshold: float) -> list[Extremum]:
    new_extremas = []
    delta_min = 0.5
    m_limit = lambda o,s: round(np.sqrt(2)*lambda_descr*s)
    gradient_2d = scale_space_gradients(scale_space, m_limit, m_limit)
    for extrema in extremas:
        limit = np.sqrt(2)*lambda_descr*extrema.sigma
        image = scale_space[extrema.o][extrema.s]
        delta_o = delta_min*2**(extrema.o)
        x = extrema.x
        y = extrema.y
        if(limit <= x and x <= image.shape[1]-limit and limit <= y and y <= image.shape[0]-limit):
            histograms = [[np.zeros(n_ori) for _ in range(n_hist)] for _ in range(n_hist)]
            
            for m in range(round((x-limit*(n_hist+1)/n_hist)/delta_o), round((x+limit*(n_hist+1)/n_hist)/delta_o)):
                for n in range(round((y-limit*(n_hist+1)/n_hist)/delta_o), round((y+limit*(n_hist+1)/n_hist)/delta_o)):
                    x_mn = ((m*delta_o - x)*np.cos(extrema.orientation) + (n*delta_o-y)*np.sin(extrema.orientation))/extrema.sigma
                    y_mn = (-(m*delta_o - x)*np.sin(extrema.orientation) + (n*delta_o-y)*np.cos(extrema.orientation))/extrema.sigma
                    
                    if(max(abs(x_mn), abs(y_mn)) < lambda_descr*(n_hist+1)/n_hist):
                        dx = gradient_2d[extrema.o,round(extrema.sigma),n,m][0]
                        dy = gradient_2d[extrema.o,round(extrema.sigma),n,m][1]
                        Delta_mn =  (np.arctan2(dx, dy) - extrema.orientation)%(2*np.pi)
                        
                        c_descr = np.exp(-(np.linalg.norm(np.array([m*delta_o, n*delta_o]) - np.array([x,y]))**2)/(2*(lambda_descr*extrema.sigma)**2))*np.linalg.norm([dx,dy])
                        for i in range(n_hist):
                            x_i = ((i+1)-(1+n_hist)/2)*(2*lambda_descr/n_hist)
                            if(abs(x_i-x_mn) > (2*lambda_descr/n_hist)):
                                continue
                            for j in range(n_hist):
                                y_j = ((j+1)-(1+n_hist)/2)*(2*lambda_descr/n_hist)
                                if(np.abs(y_j-y_mn) > (2*lambda_descr/n_hist)):
                                    continue
                                for k in range(n_ori):
                                    Delta_k = 2*np.pi*(k)/n_ori
                                    if((Delta_k-Delta_mn)%(2*np.pi) >= (2*np.pi/n_ori)):
                                        continue
                                    temp = (1-(n_hist/(2*lambda_descr))*abs(x_mn-x_i))
                                    temp *= (1-(n_hist/(2*lambda_descr))*abs(y_mn-y_j))
                                    temp *= (1-(n_ori/(2*np.pi))*np.abs((Delta_mn - Delta_k)%(2*np.pi)))
                                    temp *= c_descr
                                    
                                    histograms[i][j][k] += temp
            feature_temp = []
            for i in range(n_hist):
                for j in range(n_hist):
                    for k in range(n_ori):
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
                  extremas: list[Extremum], 
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
                       show_plots: bool = False) -> Tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[Extremum]]:
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
