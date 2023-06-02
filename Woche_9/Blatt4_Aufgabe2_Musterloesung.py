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
            m (int): The x - coordinate in Dog image - 0-based
            n (int): The y - coordinate in Dog image - 0-based
            o (int): The octave - 1-based
            s (int): The scale level from Dog image - 0-based, but no transformation to 1-based needed.
            x (int, optional): The x - coordinate in scale space - 0-based. Defaults to 0.
            y (int, optional): the y - coordinate in scale space - 0-based. Defaults to 0.
            sigma (float, optional): The scale level in scale space - 0-based, but no transformation to 1-based needed. Defaults to 0.0.
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
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, omega: {self.omega}, ori: {self.orientation}, mag: {self.magnitude}]"

    def __repr__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, omega: {self.omega}, ori: {self.orientation}, mag: {self.magnitude}]"

def create_scale_space(img: np.ndarray, octaves: int, scales: int, delta_min: float = 0.5, sigma_min: float = 0.8) -> Tuple[list[list[np.ndarray]],list[float],list[list[float]]]:
    """Creates a scale space for a given image

    Args:
        img (np.ndarray): the image
        octaves (int): the number of octaves
        scales (int): the number of scales
        delta_min (float): the starting delta value. Defaults to 0.5.
        sigma_min (float): the starting sigma value. Defaults to 0.8.

    Returns:
        Tuple[list[list[np.ndarray]],list[float],list[list[float]]]: the scale space divide into octave - scales - images, the delta values and the sigma values
    """
    # create the scale space
    scale_space: list[list[np.ndarray]] = []
    deltas: list[float] = [delta_min]
    sigmas: list[list[float]] = [[sigma_min]]
    # represent current scale factor
    delta_prev = delta_min
    # scale image up with bilinear interpolation
    u = cv.resize(img, (0,0), fx=1.0 / delta_prev, fy=1.0 / delta_prev, interpolation=cv.INTER_LINEAR)
    # first seed image blurred
    v_1_0 = cv.GaussianBlur(u, (0,0), sigma_min)
    first_octave: list[np.ndarray] = [v_1_0]
    # first octave calculation
    for s in range(1, scales+1):
        # delta_o/delta_min is 1 here
        # scales is equals to n_spo + 2
        sigma = sigma_min*2**((s/(scales-2)))
        sigmas[0].append(sigma)
        first_octave.append(cv.GaussianBlur(first_octave[s-1], (0,0), sigma))
    # remove seed image from octave
    # first_octave = first_octave[1:]
    # and append octave
    scale_space.append(first_octave)
    # for every other octave excluding the first
    for o in range(2, octaves+1):
        # delta o
        delta_o = delta_min*2**(o-1)
        deltas.append(delta_o)
        # seed image is antepenultimate image of previous octave
        # o is 1-based, index is 0-based, therefore o-2
        seed = scale_space[o-2][scales-3]
        sigmas.append([sigmas[o-2][scales-3]])
        # scale image down with bilinear interpolation
        seed = cv.resize(seed, (0,0), fx=delta_prev / delta_o, fy=delta_prev / delta_o, interpolation=cv.INTER_LINEAR)
        current_octave: list[np.ndarray] = [seed]
        # for every scale level (n_spo+2)
        for s in range(1, scales+1):
            # calculate sigma
            sigma = delta_o/delta_min * sigma_min*2**((s/(scales-2)))
            sigmas[o-1].append(sigma)
            # and apply blur to previous scale image
            current_octave.append(cv.GaussianBlur(current_octave[s-1], (0,0), sigma))
        # remove seed image from octave
        # current_octave = current_octave[1:]
        # and append octave
        scale_space.append(current_octave)
        delta_prev = delta_o
    return scale_space, deltas, sigmas

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
            dog_row.append(cv.subtract(octave[i+1], octave[i]))
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
                        # assing octave to be 1-based
                        extremas.append(Extremum(m,n,octave_index+1,scale_index))
    # results in MAX_SCALE-3 possible extrema images per octave
    # since we cannot take 1st and last image of each octave
    return extremas

def taylor_expansion(extremas: list[Extremum],
                     dog_scales: list[list[np.ndarray]],
                     drop_off: float,
                     c_dog: float,
                     deltas: list[float],
                     sigmas: list[list[float]]) -> list[Extremum]:
    """Finetunes locations of extrema using taylor expansion

    Args:
        extremas (list[Extremum]): The extremas to finetune
        dog_scales (list[list[np.ndarray]]): The difference of gaussians images to finetune with
        drop_off (float): if offset is below, we accept a new location
        c_dog (float): if dog value is below, we discard the extrema
        delta_min (float): the starting delta value. Defaults to 0.5.
        sigma_min (float): the starting sigma value. Defaults to 0.8.

    Returns:
        list[Extremum]: The new Extremum. Newly created Extremum Objects
    """
    new_extremas = []
    for extremum in extremas:
        # discard low contrast candiate keypoints
        if(np.abs(dog_scales[extremum.o-1][extremum.s][extremum.n, extremum.m]) < 0.8 * c_dog):
            continue
        # location of the extremum
        # will be adjusted maximum 5 times
        # locations are 0-based
        # attention when calculating with values
        # s is 1 based, as there exists the seed image infront
        cs = extremum.s
        cm = extremum.m
        cn = extremum.n
        # o-index must be 0-based, but is 1-based in the extremum
        o_index = extremum.o-1
        # for each adjustment
        # will break if new location is found
        for _ in range(5):
            current = dog_scales[o_index][cs]
            previous = dog_scales[o_index][cs-1]
            next_scale = dog_scales[o_index][cs+1]
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
                delta_current = deltas[o_index]
                # sigma is calculated from the scale
                #sigma = delta_current/deltas[0]*sigma[0][0]*2**((alpha[0,0]+cs)/MAX_SCALE-2)
                sigma = sigmas[o_index][round(cs+alpha[0,0])]
                # and the keypoint coordinates
                # coordinates are 0-based, therefore we add 1 to current-location and subtract 1 from total
                x = delta_current*(alpha[1,0]+cm+1)-1
                y = delta_current*(alpha[2,0]+cn+1)-1
                # create new Extremum object with the corresponding values
                new_extremum = Extremum(cm, cn, extremum.o, cs, x, y, sigma, omega)
                new_extremas.append(new_extremum)
                break
            # reject extrema if offset is beyond image border or scale border
            if(round(cs+alpha[0,0]) < 1
               or round(cs+alpha[0,0]) >= len(dog_scales[o_index])-1):
                break
            if(round(cm+alpha[1,0]) < 1
               or round(cm+alpha[1,0]) >= current.shape[1]-1):
                break
            if(round(cn+alpha[2,0]) < 1
               or round(cn+alpha[2,0]) >= current.shape[0]-1):
                break
            # if the new location is valid, update the locations in that direction
            # at least one value will be adjust (as at least one is >0.6)
            cs = round(cs + alpha[0,0])
            cm = round(cm + alpha[1,0])
            cn = round(cn + alpha[2,0])
    return new_extremas

def filter_extremas(extremas: list[Extremum], dogs: list[list[np.ndarray]], c_dog: float, c_edge: float) -> list[Extremum]:
    """Filters Extrema based on contrast and curvature

    Args:
        extremas (list[Extremum]): The extrema to filter
        dogs (list[list[np.ndarray]]): The dog values to calculate curvature from
        c_dog (float): Contrast Threshold for the interpolated dog value
        c_edge (float): Curvature Threshold for the curvature ratio

    Returns:
        list[Extremum]: Filtered Extremum. Returns same objects from input.
    """
    filtered_extremas = []
    for extremum in extremas:
        # current location of the extremum
        cs = extremum.s
        cm = extremum.m
        cn = extremum.n
        # current dog image
        current = dogs[extremum.o-1][cs]
        
        # contrast drop off from the calculate omega value of taylor expansion
        if(abs(extremum.omega) < c_dog):
            continue
        # filter off extremas at the border
        if(cm < 1 or cm > current.shape[1]-2 or cn < 1 or cn > current.shape[0]-2):
            continue
        
        # 2d-Hessian matrix
        dxx = (current[cn, cm+1] - current[cn, cm-1]) / 2
        dyy = (current[cn+1, cm]- current[cn-1, cm]) / 2
        # dxy will be resued for dyx
        dxy = (current[cn+1, cm+1] - current[cn+1, cm-1]-current[cn-1, cm+1] - current[cn-1, cm-1])/4
        # paper annotates 1 = x, 2 = y
        H = np.matrix([[dxx, dxy], [dxy, dyy]])
        
        trace = np.trace(H)
        determinant = np.linalg.det(H)
        # if we divide by 0, extremum is not valid
        if(determinant == 0):
            continue
        curvature_ratio = (trace*trace)/determinant
        
        # curvature drop off
        if(curvature_ratio >= (c_edge+1)**2/c_edge):
            continue
        
        filtered_extremas.append(extremum)
    return filtered_extremas

def get_scale_space_gradient_2d(scale_space: list[list[np.ndarray]], octave: int, scale: int, n: int, m: int) -> Tuple[float, float]:
    image = scale_space[octave-1][scale]
    return [((image[n,m+1]-image[n,m-1])/2),((image[n+1,m]-image[n-1,m])/2)]
def assign_orientations(extremas: list[Extremum], 
                        scale_space: list[list[np.ndarray]], 
                        lambda_ori: float, 
                        n_bins: int, 
                        deltas: list[float]) -> list[Extremum]:
    """Assign orientation angle and magnitude to each Keypoint.

    Args:
        extremas (list[Extremum]): List of extrema to assign orientation to
        scale_space (list[list[np.ndarray]]): The scale space to calculate from
        lambda_ori (float): window size, calculated by 3*lambda_ori*sigma of the current extremum
        num_bins (int): number of bins per histogram. 36 is recommended.
        deltas (list[float]): list of deltas for each octave

    Returns:
        list[Extremum]: List of extrema with assigned orientation. Returns new Extremum objects.
    """
    # create gaussian kernels for each scale
    new_extremas = []
    
    for extremum in extremas:
        # keypoint coordinates
        # octave is 1-based
        o = extremum.o
        # coordinates are 0 based
        o_index = o-1
        x = extremum.x
        y = extremum.y
        # current delta
        delta_o = deltas[o_index]
        # currrent window size
        limit = 3*lambda_ori*extremum.sigma
        # current image in scale space
        image = scale_space[o_index][extremum.s]
        # filter out extremas too close to edge
        # where window size does not fit in image
        lower_x_limit = round((x+1-limit)/delta_o)
        upper_x_limit = round((x+1+limit)/delta_o)
        lower_y_limit = round((y+1-limit)/delta_o)
        upper_y_limit = round((y+1+limit)/delta_o)
        if(lower_x_limit < 0 or upper_x_limit >= image.shape[1] or lower_y_limit < 0 or upper_y_limit >= image.shape[0]):
            continue
        # initialize histogram
        histogram = np.zeros(n_bins)
        # for each pixel in the window
        for m in range(lower_x_limit, upper_x_limit+1):
            for n in range(lower_y_limit, upper_y_limit+1):
                # get current gradient
                gradient = get_scale_space_gradient_2d(scale_space, o, extremum.s,n-1,m-1)
                dx = gradient[0]
                dy = gradient[1]
                # calculate added gaussian weight
                c_ori = np.exp(-np.linalg.norm(np.array([m*delta_o, n*delta_o])-np.array([x,y]))**2/(2*(lambda_ori*delta_o)**2))*np.linalg.norm([dx,dy])
                # calculate bin index from given gradients
                bin_index = round((n_bins/(2*np.pi))*(np.arctan2(dx,dy)%(2*np.pi)))-1
                # add to histogram
                histogram[bin_index] += c_ori
        # smooth histogram 6 times by box convolution
        for _ in range(6):
            histogram = signal.convolve(histogram, np.array([1,1,1])/3)
        # find 80% max value
        max_value = np.max(histogram)*0.8
        # for each histogram bin
        for k, h_k in enumerate(histogram):
            # get previous and next bin
            prev_hist = histogram[(k-1)%n_bins]
            next_hist = histogram[(k+1)%n_bins]
            # if current bin is top 80% and local maximum
            if(h_k > prev_hist and h_k > next_hist and h_k > max_value):
                # delta_k takes k-1 as value, but k is 0-based in loop
                # therefore k_1_based-1 = k_0_based
                Delta_k = 2*np.pi*(k)/n_bins
                # calculate orientation
                Delta_key = Delta_k+np.pi/n_bins*((prev_hist - next_hist)/(prev_hist + next_hist - 2*h_k))
                # create new extremum for each bin passing the above criteria
                new_extrema = Extremum(extremum.m, extremum.n, extremum.o, extremum.s, x, y, extremum.sigma, extremum.omega, Delta_key, h_k)
                new_extremas.append(new_extrema)
    return new_extremas

def create_descriptors(extremas: list[Extremum],
                       scale_space: list[list[np.ndarray]],
                       lambda_descr: float,
                       n_hist: int,
                       n_ori: int,
                       gradient_threshold: float,
                       deltas: list[float]) -> list[Extremum]:
    """Creates Key Descriptors for each Keypoint.

    Args:
        extremas (list[Extremum]): The keypoints to create descriptors for
        scale_space (list[list[np.ndarray]]): The scalespace to calculate from
        lambda_descr (float): size of the window, calculated by 6*lambda_descr*sigma of the current extremum
        n_hist (int): number of histograms in that window
        n_ori (int): number of bins per histogram
        gradient_threshold (float): caps the feature vector at gradient_threshold*norm(feature_vector)
        deltas (list[float]): list of deltas for each octave

    Returns:
        list[Extremum]: The keypoints with descriptors. Extrema are same objects.
    """
    new_extremas = []
    # Window size calculated from lambda_descr and scale
    #get_limit = lambda o,s: round(2*lambda_descr*((n_hist+1)/n_hist)*s)
    # calculate gradients for each scale
    #gradient_2d = scale_space_gradients(scale_space, get_limit)
    for extremum in extremas:
        # current window size
        limit = lambda_descr*(n_hist+1)/n_hist*extremum.sigma
        # current image, octave is 1-based
        image = scale_space[extremum.o-1][extremum.s]
        delta_o = deltas[extremum.o-1]
        
        x = extremum.x
        y = extremum.y
        lower_x_limit = round((x-limit*(n_hist+1)/n_hist)/delta_o)
        upper_x_limit = round((x+limit*(n_hist+1)/n_hist)/delta_o)
        lower_y_limit = round((y-limit*(n_hist+1)/n_hist)/delta_o)
        upper_y_limit = round((y+limit*(n_hist+1)/n_hist)/delta_o)
        # if keypoint is not too close to edge
        if(lower_x_limit < 0 or upper_x_limit >= image.shape[1] or lower_y_limit < 0 or upper_y_limit >= image.shape[0]):
            continue
        # initialize histograms
        histograms = [[np.zeros(n_ori) for _ in range(n_hist)] for _ in range(n_hist)]
        # for each point in the window
        for m in range(lower_x_limit, upper_x_limit+1):
            for n in range(lower_y_limit, upper_y_limit+1):
                # normalized coordinates
                x_mn = ((m*delta_o - x)*np.cos(extremum.orientation) + (n*delta_o-y)*np.sin(extremum.orientation))/extremum.sigma
                y_mn = (-(m*delta_o - x)*np.sin(extremum.orientation) + (n*delta_o-y)*np.cos(extremum.orientation))/extremum.sigma
                # verify that point is in normalized window
                if(max(abs(x_mn), abs(y_mn)) < lambda_descr*(n_hist+1)/n_hist):
                    # get gradients
                    gradient = get_scale_space_gradient_2d(scale_space, extremum.o, extremum.s, n-1, m-1)
                    dx = gradient[0]
                    dy = gradient[1]
                    # calculate nrormalized gradient orientation
                    # by subtracting the keypoint orientation
                    Delta_mn =  (np.arctan2(dx, dy) - extremum.orientation)%(2*np.pi)
                    # calculate total contribution to histogram
                    c_descr = np.exp(-(np.linalg.norm(np.array([m*delta_o, n*delta_o]) - np.array([x,y]))**2)/(2*(lambda_descr*extremum.sigma)**2))*np.linalg.norm([dx,dy])
                    # for each histogram
                    for i in range(n_hist):
                        # if corresponding x coordinate is not in histogram
                        x_i = ((i+1)-(1+n_hist)/2)*(2*lambda_descr/n_hist)
                        # skip if not in histogram
                        if(abs(x_i-x_mn) > (2*lambda_descr/n_hist)):
                            continue
                        for j in range(n_hist):
                            y_j = ((j+1)-(1+n_hist)/2)*(2*lambda_descr/n_hist)
                            if(np.abs(y_j-y_mn) > (2*lambda_descr/n_hist)):
                                continue
                            # for each bin
                            for k in range(n_ori):
                                # if corresponding orientation is not in bin
                                Delta_k = 2*np.pi*(k)/n_ori
                                if((Delta_k-Delta_mn)%(2*np.pi) >= (2*np.pi/n_ori)):
                                    continue
                                # calculate new histogram value
                                temp = (1-(n_hist/(2*lambda_descr))*abs(x_mn-x_i))
                                temp *= (1-(n_hist/(2*lambda_descr))*abs(y_mn-y_j))
                                temp *= (1-(n_ori/(2*np.pi))*np.abs((Delta_mn - Delta_k)%(2*np.pi)))
                                temp *= c_descr
                                
                                histograms[i][j][k] += temp
        # create feature vector
        feature_temp = []
        for i in range(n_hist):
            for j in range(n_hist):
                for k in range(n_ori):
                    feature_temp.append(histograms[i][j][k])
        feature_vector = np.array(feature_temp)
        # and calculate normalized feature vector
        normalized = np.linalg.norm(feature_vector)
        # for each element in feature vector
        for i in range(len(feature_vector)):
            # if value is too high, cap it
            feature_vector[i] = min(feature_vector[i], gradient_threshold*normalized)
            # and quantizie to 8bit integers
            feature_vector[i] = min(np.floor(512*feature_vector[i]/normalized), 255)
        extremum.descriptor = feature_vector
        new_extremas.append(extremum)
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
    plt.scatter([extremum.m for extremum in EXTREMAS 
                    if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET], 
                [extremum.n for extremum in EXTREMAS 
                    if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET], c="r", s=1)
    # draw orientations
    if(SHOW_ORIENTATIONS):
        plt.quiver([extremum.m for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET],
                   [extremum.n for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET],
                   [np.sin(extremum.orientation*np.pi/180)*extremum.magnitude for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET],
                   [np.cos(extremum.orientation*np.pi/180)*extremum.magnitude for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET])
    ## print descriptors
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE + SCALE_OFFSET}")
        print(np.matrix([extremum.descriptor for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE + SCALE_OFFSET]))
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
    event.canvas.figure.gca().scatter([extremum.m for extremum in EXTREMAS
                                       if extremum.o-1 == CURRENT_OCTAVE and extremum.s == CURRENT_SCALE + SCALE_OFFSET],
                                      [extrema.n for extrema in EXTREMAS
                                       if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET], c="r", s=1)
    if(SHOW_ORIENTATIONS):
        event.canvas.figure.gca().quiver(   [extrema.m for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [extrema.n for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [np.sin(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET],
                                            [np.cos(extrema.orientation*np.pi/180)*extrema.magnitude for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE + SCALE_OFFSET])
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE + SCALE_OFFSET}")
        print(np.matrix([extremum.descriptor for extremum in EXTREMAS if extremum.o-1 == CURRENT_OCTAVE and extremum.s == CURRENT_SCALE + SCALE_OFFSET]))
    event.canvas.draw()
    
def detect_and_compute(img: np.ndarray, 
                       taylor_threshold: float = 0.6,
                       c_dog: float = 0.015,
                       c_edge: float = 10.0,
                       lambda_ori: float = 1.5,
                       n_bins: int = 36,
                       lambda_descr: float = 6,
                       n_hist: int = 4,
                       n_ori: int = 8,
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
        lambda_ori (float, optional): Window Size of orientation calculation. Size equates to 6*lambda_ori*sigma. Defaults to 1.5.
        n_bins (int, optional): Number of Histogram bins for the orientation calculation. Should divide into 360. Defaults to 36.
        lambda_descr (float, optional): Area around each extrema, for which descriptors are calculated.
        Size equates 2*lambda_descr*(n_hist+1)/n_hist*sigma. Defaults to 16.
        n_hist (int, optional): number of histograms per axis. Defaults to 4.
        n_ori (int, optional): Number of histogram bins for each descriptor. Should divide into 360. Defaults to 8.
        descriptor_gradient_threshold (float, optional): Histogram directions below this are set to 0 for each descriptor. Defaults to 0.2.
        show_plots (bool, optional): Shows Matplotlib plots for each step. Defaults to False.

    Returns:
        Tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[Extrema]]: scale space [octave][scale], dogs [octave, scale-1], extremas
    """
    scale_space, deltas, sigmas = create_scale_space(img, MAX_OCTAVE, MAX_SCALE)
    # show_array(scale_space, "Scale Space")
    dogs = create_dogs(scale_space)
    normalized_dogs = [[cv.normalize(d, None, 0, 1, cv.NORM_MINMAX) for d in octave] for octave in dogs] # type: ignore
    # show_array(dogs, "Normalized DoGs")
    discrete_extremas = find_discrete_extremas(dogs)
    # show_extremas(dogs, discrete_extremas, "Discrete Extremas")
    taylor_extremas = taylor_expansion(discrete_extremas,
                                       dogs,
                                       taylor_threshold,
                                       c_dog,
                                       deltas,
                                       sigmas)
    # show_extremas(dogs, taylor_extremas, "Taylor Extremas")
    filtered_extremas = filter_extremas(taylor_extremas,
                                        dogs, c_dog,
                                        c_edge)
    # show_extremas(dogs, filtered_extremas, "Filtered Extremas")
    key_points = assign_orientations(filtered_extremas,
                                     scale_space,
                                     lambda_ori,
                                     n_bins,
                                     deltas)
    # show_extremas(scale_space, key_points, "Key Points", 0, 2, MAX_SCALE-1, True)
    descriptor_key_points = create_descriptors(key_points,
                                               scale_space,
                                               lambda_descr,
                                               n_hist,
                                               n_ori,
                                               descriptor_gradient_threshold,
                                               deltas)
    # show_extremas(scale_space, descriptor_key_points, "Descriptor Key Points", 0, 2, MAX_SCALE-1, True, True)
    
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
    gray = (gray/255.).astype(np.float32)
    rotated = rotate_image(gray, 45)
    scale_space, normalized_dogs, key_points = detect_and_compute(gray, show_plots=True)
    rotated_scale_space, rotated_normalized_dogs, roated_key_points = detect_and_compute(rotated, show_plots=True)

    # Keypoint Matching
    
if __name__ == "__main__":
    main()
