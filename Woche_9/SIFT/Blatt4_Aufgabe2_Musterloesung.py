import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable
from scipy import signal
import scipy

from KeyPoint import KeyPoint
from SIFT_Params import SIFT_Params
        


def create_scale_space(u_in: NDArray[np.float32], sift_params: SIFT_Params) -> Tuple[list[list[NDArray[np.float32]]],list[float],list[list[float]]]:
    """Creates a scale space for a given image

    Args:
        u_in (NDArray[np.float32]): the image
        sift_Params (SIFT_Params): the sift parameters

    Returns:
        Tuple[list[list[NDArray[np.float32]]],list[float],list[list[float]]]: the scale space divide into octave - scales - images, the delta values and the sigma values
    """
    # Variable Initialization
    scale_space: list[list[np.ndarray]] = []
    deltas: list[float] = [sift_params.delta_min]
    sigmas: list[list[float]] = [[sift_params.sigma_min]]
    delta_prev = sift_params.delta_min
    # scale image up with bilinear interpolation
    u = cv.resize(u_in, (0,0), fx=1.0 / sift_params.delta_min, fy=1.0 / sift_params.delta_min, interpolation=cv.INTER_LINEAR)
    # first seed image blurred
    v_1_0 = cv.GaussianBlur(u, (0,0), sift_params.sigma_min)
    first_octave: list[np.ndarray] = [v_1_0]
    
    # first octave calculation
    # delta_o equates to delta_min here
    delta_o = sift_params.delta_min*2**(1-1)
    for s in range(1, sift_params.n_spo+3):
        sigma = delta_o/sift_params.delta_min * sift_params.sigma_min*2**(s/sift_params.n_spo)
        sigmas[0].append(sigma)
        first_octave.append(cv.GaussianBlur(first_octave[s-1], (0,0), sigma))
    # and append octave
    scale_space.append(first_octave)
    
    # for every other octave excluding the first
    for o in range(2, sift_params.n_oct+1):
        # delta o
        delta_o = sift_params.delta_min*2**(o-1)
        deltas.append(delta_o)
        # seed image is antepenultimate image of previous octave
        # o is 1-based, index is 0-based, therefore o-2
        # n_spo is 1-based, index is 0-based, therefore n_spo-1
        seed = scale_space[o-2][sift_params.n_spo-1]
        sigmas.append([sigmas[o-2][sift_params.n_spo-1]])
        # scale image down with bilinear interpolation
        seed = cv.resize(seed, (0,0), fx=delta_prev / delta_o, fy=delta_prev / delta_o, interpolation=cv.INTER_LINEAR)
        current_octave: list[np.ndarray] = [seed]
        # for every scale level (n_spo+2)
        for s in range(1, sift_params.n_spo+3):
            # calculate sigma
            sigma = delta_o/sift_params.delta_min * sift_params.sigma_min*2**(s/sift_params.n_spo)
            sigmas[o-1].append(sigma)
            # and apply blur to previous scale image
            current_octave.append(cv.GaussianBlur(current_octave[s-1], (0,0), sigma))
        # and append octave
        scale_space.append(current_octave)
        delta_prev = delta_o
    return scale_space, deltas, sigmas

def create_dogs(scale_space: list[list[NDArray[np.float32]]], sift_params: SIFT_Params) -> list[list[NDArray[np.float32]]]:
    """Creates the difference of gaussians for a given scale space

    Args:
        scale_space (list[list[NDArray[np.float32]]]): the scale space
        sift_params (SIFT_Params): the sift parameters

    Returns:
        list[list[NDArray[np.float32]]]: the difference of gaussians
    """
    dogs = []
    # for each octave
    for o in range(1, sift_params.n_oct+1):
        octave = scale_space[o-1] # 0-based index
        dog_row = []
        # for each scale level (excluding last image)
        for s in range(sift_params.n_spo+2):
            # calculate difference of gaussians
            dog_row.append(cv.subtract(octave[s+1], octave[s]))
        dogs.append(dog_row)
    return dogs

def find_discrete_extremas(dogs: list[list[NDArray[np.float32]]], sift_params: SIFT_Params) -> list[KeyPoint]:
    """Finds the discrete extremas for a given difference of gaussians

    Args:
        dogs (list[list[NDArray[np.float32]]]): the difference of gaussians
        sift_params (SIFT_Params): the sift parameters (used for n_oct and n_spo)

    Returns:
        list[KeyPoint]: the discrete extremas
    """
    extremas = []
    # for each octave in dogs
    for o in range(1, sift_params.n_oct+1):
        print(f"Extrema Calculation: Octave {o}")
        octave = dogs[o-1] # 0-based index
        # for each dog image in the octave excluding first and last image
        for s in range(1, sift_params.n_spo+1):
            # the current dog image
            current = octave[s]
            # the one before
            before = octave[s-1]
            # the one after
            after = octave[s+1]
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
                        extremas.append(KeyPoint(o,s,m,n))
    # results in MAX_SCALE-3 possible extrema images per octave
    # since we cannot take 1st and last image of each octave
    return extremas

def taylor_expansion(extremas: list[KeyPoint],
                     dog_scales: list[list[NDArray[np.float32]]],
                     sift_params: SIFT_Params,
                     deltas: list[float],
                     sigmas: list[list[float]]) -> list[KeyPoint]:
    """Finetunes locations of extrema using taylor expansion

    Args:
        extremas (list[KeyPoint]): The extremas to finetune
        dog_scales (list[list[NDArray[np.float32]]]): The difference of gaussians images to finetune with
        sift_params (SIFT_Params): The sift parameters
        deltas (list[float]): The deltas for each octave
        sigmas (list[list[float]]): The sigmas for each octave and scale level
    Returns:
        list[KeyPoint]: The new Extremum. Newly created KeyPoint Objects
    """
    new_extremas = []
    for extremum in extremas:
        # discard low contrast candiate keypoints
        if(abs(dog_scales[extremum.o-1][extremum.s][extremum.n, extremum.m]) < 0.8 * sift_params.C_DoG):
            continue
        # locations are 0-based
        # attention when calculating with values
        # s is 1 based, as there exists the seed image infront
        s = extremum.s
        m = extremum.m
        n = extremum.n
        # for each adjustment
        # will break if new location is found
        # will be adjusted maximum 5 times
        for _ in range(5):
            current = dog_scales[extremum.o-1][s]
            previous = dog_scales[extremum.o-1][s-1]
            next_scale = dog_scales[extremum.o-1][s+1]
            # called $\bar{g}^o_{s,m,n}$ in the paper
            # represent the first derivative  in a finite difference scheme
            # is Transposed, as we calculate [a,b,c] values, but want [[a],[b],[c]]
            g_o_smn = np.matrix([[next_scale[n, m] - previous[n, m], current[n, m+1] - current[n, m-1], current[n+1, m]- current[n-1, m]]]).T
            
            # calcuation of hessian matrix
            h11 = (next_scale[n, m] - previous[n, m] - 2*current[n, m])
            h22 = (current[n, m+1] + current[n, m-1] - 2*current[n, m])
            h33 = (current[n+1, m]- current[n-1, m] - 2*current[n, m])
            
            # h12, h13 and h23 are reused for h21, h31 and h32
            # as they are the same value
            h12 = (next_scale[n, m+1] - next_scale[n, m-1]- previous[n, m+1] - previous[n, m-1])/4
            h13 = (next_scale[n+1, m] - next_scale[n-1, m]- previous[n+1, m] - previous[n-1, m])/4
            h23 = (current[n+1, m+1] - current[n+1, m-1]-current[n-1, m+1] - current[n-1, m-1])/4
            
            hessian = np.matrix([[h11, h12, h13],
                                 [h12, h22, h23], 
                                 [h13, h23, h33]])
            # inverse of a matrix with det = 0 is not possible
            # therefore we break here
            if(np.linalg.det(hessian) == 0):
                break
            # calculate offset
            alpha = -np.linalg.inv(hessian) * g_o_smn
            # the every value is below the drop_off
            # we found the new location
            if(np.max(np.abs(alpha)) < 0.6):
                if(round(s+alpha[0,0]) < 1
                    or round(s+alpha[0,0]) >= len(dog_scales[extremum.o-1])-1):
                    break
                if(round(m+alpha[1,0]) < 1
                    or round(m+alpha[1,0]) >= current.shape[1]-1):
                    break
                if(round(n+alpha[2,0]) < 1
                    or round(n+alpha[2,0]) >= current.shape[0]-1):
                    break
                s = round(s + alpha[0,0])
                m = round(m + alpha[1,0])
                n = round(n + alpha[2,0])
                # this is simplified from 'w+alphaT*g + 0.5*alphaT*H*alpha'
                # to 'w+0.5*alphaT*g' following the paper
                # pseudocode does not simplify here
                # omega represent the value of the DoG interpolated extremum
                omega = current[extremum.n, extremum.m] + 0.5*alpha.T*g_o_smn
                # calculate the current delta and sigma for the corresponding new location
                delta_oe = deltas[extremum.o-1]
                # sigma is calculated from the scale
                sigma = (delta_oe/sift_params.delta_min)*sift_params.sigma_min*2**((alpha[0,0]+s)/sift_params.n_spo)
                # and the keypoint coordinates
                x = delta_oe*(alpha[1,0]+m)
                y = delta_oe*(alpha[2,0]+n)
                # create new Extremum object with the corresponding values
                new_extremum = KeyPoint(extremum.o, s, m, n, sigma, x, y, omega)
                new_extremas.append(new_extremum)
                break
            # reject extrema if offset is beyond image border or scale border
            if(round(s+alpha[0,0]) < 1
              or round(s+alpha[0,0]) >= len(dog_scales[extremum.o-1])-1):
               break
            if(round(m+alpha[1,0]) < 1
              or round(m+alpha[1,0]) >= current.shape[1]-1):
               break
            if(round(n+alpha[2,0]) < 1
              or round(n+alpha[2,0]) >= current.shape[0]-1):
               break
            # if the new location is valid, update the locations in that direction
            # at least one value will be adjust (as at least one is >0.6)
            s = round(s + alpha[0,0])
            m = round(m + alpha[1,0])
            n = round(n + alpha[2,0])
    return new_extremas

def filter_extremas(extremas: list[KeyPoint], dogs: list[list[NDArray[np.float32]]], sift_params: SIFT_Params) -> list[KeyPoint]:
    """Filters Extrema based on contrast and curvature

    Args:
        extremas (list[KeyPoint]): The extrema to filter
        dogs (list[list[NDArray[np.float32]]]): The dog values to calculate curvature from
        sift_params (SIFT_Params): The sift parameters

    Returns:
        list[KeyPoint]: Filtered Extremum. Returns same objects from input.
    """
    filtered_extremas = []
    for extremum in extremas:
        # current location of the extremum
        s = extremum.s
        m = extremum.m
        n = extremum.n
        # current dog image
        current = dogs[extremum.o-1][s]
        
        # contrast drop off from the calculate omega value of taylor expansion
        if(abs(extremum.omega) < sift_params.C_DoG):
            continue
        # filter off extremas at the border
        if(m < 1 or m > current.shape[1]-2 or n < 1 or n > current.shape[0]-2):
            continue
        
        # 2d-Hessian matrix
        h11 = (current[n, m+1] - current[n, m-1]) / 2
        h22 = (current[n+1, m]- current[n-1, m]) / 2
        # h12 will be resued for h21
        h12 = (current[n+1, m+1] - current[n+1, m-1]-current[n-1, m+1] - current[n-1, m-1])/4
        hessian = np.matrix([[h11, h12],
                             [h12, h22]])
        
        trace = np.trace(hessian)
        determinant = np.linalg.det(hessian)
        # if we divide by 0, extremum is not valid
        if(determinant == 0):
            continue
        edgeness = (trace*trace)/determinant
        
        # curvature drop off
        if(edgeness >= ((sift_params.C_edge+1)**2)/sift_params.C_edge):
            continue
        
        filtered_extremas.append(extremum)
    return filtered_extremas

def get_scale_space_gradient_2d(scale_space: list[list[np.ndarray]], octave: int, scale: int, n: int, m: int) -> Tuple[float, float]:
    image = scale_space[octave-1][scale]
    return [((image[n,m+1]-image[n,m-1])/2),((image[n+1,m]-image[n-1,m])/2)]


def gradient_2d(scale_space: list[list[NDArray[np.float32]]], sift_params: SIFT_Params) -> dict[Tuple[int, int, int, int], Tuple[float, float]]:
    gradients: dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
    for o in range(1, sift_params.n_oct+1):
        for s in range(1, sift_params.n_spo+1):
            v_o_s = scale_space[o-1][s]
            for m in range(1, v_o_s.shape[1]-1):
                for n in range(1, v_o_s.shape[0]-1):
                    delta_m = (v_o_s[n, m+1]-v_o_s[n, m-1])/2
                    delta_n = (v_o_s[n+1, m]-v_o_s[n-1, m])/2
                    gradients[(o, s, m, n)] = (delta_m, delta_n)
    return gradients


def find_nearest_scale(keypoint: KeyPoint,
                       sigmas: list[list[float]],
                       sift_params: SIFT_Params) -> Tuple[int, int]:
    min_scale = float("inf")
    min_o = -1
    min_s = -1
    for o in range(1, sift_params.n_oct+1):
        for s in range(1, sift_params.n_spo+1):
            if(abs(sigmas[o-1][s] - keypoint.sigma) < min_scale):
                min_scale = sigmas[o-1][s]
                min_o = o
                min_s = s
    return min_o, min_s

def assign_orientations(keypoints: list[KeyPoint], 
                        scale_space: list[list[NDArray[np.float32]]],
                        sift_params: SIFT_Params,
                        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                        deltas: list[float],
                        sigmas: list[list[float]]) -> list[KeyPoint]:
    """Assign orientation angle and magnitude to each Keypoint.

    Args:
        keypoints (list[KeyPoint]): List of keypoints to assign orientation to
        scale_space (list[list[NDArray[np.float32]]]): The scale space to calculate from
        sift_params (SIFT_Params): The sift parameters
        gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
        deltas (list[float]): list of deltas for each octave

    Returns:
        list[KeyPoint]: List of keypoints with assigned orientation. KeyPoint are same objects.
    """
    # create gaussian kernels for each scale
    new_keypoints = []
    
    for keypoint in keypoints:
        
        border_limit = 3*sift_params.lambda_ori*keypoint.sigma
        nearest_o, nearest_s = find_nearest_scale(keypoint, sigmas, sift_params)
        image = scale_space[nearest_o-1][nearest_s]
        if(border_limit <= keypoint.x and
           keypoint.x <= image.shape[1]-border_limit and
           border_limit <= keypoint.y and
           keypoint.y <= image.shape[0]-border_limit):
            h_k = np.zeros(sift_params.n_bins)
            delta_o = deltas[nearest_o-1]
            for m in range(round((keypoint.x-border_limit)/delta_o), round((keypoint.x+border_limit)/delta_o)):
                for n in range(round((keypoint.y-border_limit)/delta_o), round((keypoint.y+border_limit)/delta_o)):
                    # calculate gradient magnitude and angle
                    delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                    magnitude = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                    # calculate gaussian weight
                    weight = np.exp(-np.sqrt((m*delta_o-keypoint.x)**2-(n*delta_o-keypoint.y)**2)/(2*(sift_params.lambda_ori*keypoint.sigma)**2))
                    angle = np.arctan2(delta_m, delta_n)
                    # calculate histogram index
                    # histogram is 0-based, but calculation is 1-based
                    b_index = round(sift_params.n_bins/(2*np.pi)*(angle%(2*np.pi)))-1
                    # add weight to histogram
                    h_k[b_index] += weight*magnitude
            # smooth histogram
            for _ in range(6):
                h_k = scipy.ndimage.convolve1d(h_k, weights=[1/3,1/3,1/3], mode='wrap')
            # extract reference orientation
            max_h_k = np.max(h_k)
            for k in range(sift_params.n_bins):
                prev_hist = h_k[(k-1)%sift_params.n_bins]
                next_hist = h_k[(k+1)%sift_params.n_bins]
                if(h_k[k] > prev_hist and h_k[k] > next_hist and h_k >= sift_params.t*max_h_k):
                    theta_k = (2*np.pi*k)/sift_params.n_bins
                    angle = theta_k+(np.pi/sift_params.n_bins)*((prev_hist-next_hist)/(prev_hist-2*h_k[k]+next_hist))
                    keypoint.theta = angle
                    new_keypoints.append(keypoint)
    return new_keypoints



def create_descriptors(keypoints: list[KeyPoint],
                       scale_space: list[list[NDArray[np.float32]]],
                       sift_params: SIFT_Params,
                       gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                       deltas: list[float],
                       sigmas: list[list[float]]) -> list[KeyPoint]:
    """Creates Key Descriptors for each Keypoint.

    Args:
        extremas (list[KeyPoint]): The keypoints to create descriptors for
        scale_space (list[list[NDArray[np.float32]]]): The scalespace to calculate from
        sift_params (SIFT_Params): The sift parameters
        gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
        deltas (list[float]): list of deltas for each octave

    Returns:
        list[KeyPoint]: The keypoints with descriptors. KeyPoint are same objects.
    """
    new_keypoints = []
    # Window size calculated from lambda_descr and scale
    #get_limit = lambda o,s: round(2*lambda_descr*((n_hist+1)/n_hist)*s)
    # calculate gradients for each scale
    #gradient_2d = scale_space_gradients(scale_space, get_limit)
    for keypoint in keypoints:
        
        border_limit = np.sqrt(2)*sift_params.lambda_descr*keypoint.sigma
        nearest_o, nearest_s = find_nearest_scale(keypoint, sigmas, sift_params)
        image = scale_space[nearest_o-1][nearest_s]
        if(border_limit <= keypoint.x and
           keypoint.x <= image.shape[1]-border_limit and
           border_limit <= keypoint.y and
           keypoint.y <= image.shape[0]-border_limit):
            delta_o = deltas[nearest_o-1]
            histograms = [[[np.zeros(sift_params.n_ori)] for _ in range(sift_params.n_hist)] for _ in range(sift_params.n_hist)]
            for m in range(round((keypoint.x-border_limit*(sift_params.n_hist+1)/sift_params.n_hist)/delta_o), round((keypoint.x+border_limit*(sift_params.n_hist+1)/sift_params.n_hist)/delta_o)):
                for n in range(round((keypoint.y-border_limit*(sift_params.n_hist+1)/sift_params.n_hist)/delta_o), round((keypoint.y+border_limit*(sift_params.n_hist+1)/sift_params.n_hist)/delta_o)):
                    x_vedge_mn = ((m*delta_o-keypoint.x)*np.cos(keypoint.theta)+(n*delta_o-keypoint.y)*np.sing(keypoint.theta))/keypoint.sigma
                    y_vedge_mn = (-(m*delta_o-keypoint.x)*np.sin(keypoint.theta)+(n*delta_o-keypoint.y)*np.cos(keypoint.theta))/keypoint.sigma
                    
                    if(max(abs(x_vedge_mn),abs(y_vedge_mn)) < sift_params.lambda_descr*(sift_params.n_hist+1)/sift_params.n_hist):
                        delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                        theta_mn = np.arctan2(delta_m, delta_n) - keypoint.theta%(2*np.pi)
                        magnitude = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                        weight = np.exp(-np.sqrt((m*delta_o-keypoint.x)**2-(n*delta_o-keypoint.y)**2)/(2*(sift_params.lambda_descr*keypoint.sigma)**2))
                        
                        for i  in range(sift_params.n_hist):
                            x_i = (i-(sift_params.n_hist+1)/2)*2*sift_params.lambda_descr/sift_params.n_hist
                            if(abs(x_i-x_vedge_mn)<=(2*sift_params.lambda_descr/sift_params.n_hist)):
                                for j in range(sift_params.n_hist):
                                    y_j = (j-(sift_params.n_hist+1)/2)*2*sift_params.lambda_descr/sift_params.n_hist
                                    if(abs(y_j-y_vedge_mn)<=(2*sift_params.lambda_descr/sift_params.n_hist)):
                                        for k in range(sift_params.n_ori):
                                            theta_vedge_k = (2*np.pi*k)/sift_params.n_ori
                                            if(abs(theta_vedge_k-theta_mn%(2*np.pi))<=(2*np.pi/sift_params.n_ori)):
                                                temp = weight*magnitude
                                                temp *= (1-(sift_params.n_hist)/(2*sift_params.lambda_descr)*abs(x_vedge_mn-x_i))
                                                temp *= (1-(sift_params.n_hist)/(2*sift_params.lambda_descr)*abs(y_vedge_mn-y_j))
                                                temp *= (1-(sift_params.n_ori)/(2*np.pi)*abs(theta_mn-theta_vedge_k%(2*np.pi)))
                                                histograms[i][j][k] += temp
            f = np.matrix(histograms).flatten()
            f_norm = np.linalg.norm(f)
            for l in range(1,f.shape[0]):
                f[l] = min(f[l], 0.2*f_norm)
                f[l] = min(np.floor(512*f[l]/f_norm), 255)
            keypoint.descriptor = f
            new_keypoints.append(keypoint)
    return new_keypoints

EXTREMAS = []
SCALES = []
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
                  extremas: list[KeyPoint],
                  title: str,
                  scale_end: int,
                  scale_begin = 1,
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
    
    CURRENT_OCTAVE = 0
    CURRENT_SCALE = scale_begin
    SCALE_BEGIN = scale_begin
    SCALE_END = scale_end
    SCALES = scale_space
    EXTREMAS = extremas
    SHOW_ORIENTATIONS = show_orientations
    SHOW_DESCRIPTORS = show_descriptors
    
    fig = plt.figure(num=title)
    cid = fig.canvas.mpl_connect('button_press_event', show_extrema_onclick)
    # draw image
    plt.title(f"Octave {0} Scale {SCALE_BEGIN}")
    plt.imshow(SCALES[0][SCALE_BEGIN], cmap="gray")
    # draw extremas
    plt.scatter([extremum.m for extremum in EXTREMAS 
                    if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE], 
                [extremum.n for extremum in EXTREMAS 
                    if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE], c="r", s=1)
    # draw orientations
    if(SHOW_ORIENTATIONS):
        plt.quiver([extremum.m for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE],
                   [extremum.n for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE],
                   [np.cos(extremum.theta*np.pi/180)*extremum.magnitude for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE],
                   [np.sin(extremum.theta*np.pi/180)*extremum.magnitude for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE])
    ## print descriptors
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE}")
        print(np.matrix([extremum.descriptor for extremum in EXTREMAS if extremum.o-1 == 0 and extremum.s == CURRENT_SCALE]))
    plt.show()
    fig.canvas.mpl_disconnect(cid)

def show_extrema_onclick(event):
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
    event.canvas.figure.gca().scatter([extremum.m for extremum in EXTREMAS
                                       if extremum.o-1 == CURRENT_OCTAVE and extremum.s == CURRENT_SCALE],
                                      [extrema.n for extrema in EXTREMAS
                                       if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE], c="r", s=1)
    if(SHOW_ORIENTATIONS):
        event.canvas.figure.gca().quiver(   [extrema.m for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE],
                                            [extrema.n for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE],
                                            [np.cos(extrema.theta*np.pi/180)*extrema.magnitude for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE],
                                            [np.sin(extrema.theta*np.pi/180)*extrema.magnitude for extrema in EXTREMAS
                                             if extrema.o-1 == CURRENT_OCTAVE and extrema.s == CURRENT_SCALE])
    if(SHOW_DESCRIPTORS):
        print(f"Octave: {CURRENT_OCTAVE}, extremum: {CURRENT_SCALE}")
        print(np.matrix([extremum.descriptor for extremum in EXTREMAS if extremum.o-1 == CURRENT_OCTAVE and extremum.s == CURRENT_SCALE]))
    event.canvas.draw()
    
def detect_and_compute(img: NDArray[np.float32],
                       sift_params: SIFT_Params,
                       show_plots: bool = False) -> Tuple[list[list[NDArray[np.float32]]], list[list[NDArray[np.float32]]], list[KeyPoint]]:
    """detects and computes keypoints and descriptors for a given image

    Args:
        img (NDArray[np.float32]): the image to detect keypoints and descriptors for
        sift_params (SIFT_Params): the parameters for the SIFT algorithm
        show_plots (bool, optional): Shows Matplotlib plots for each step. Defaults to False.

    Returns:
        Tuple[list[list[NDArray[np.float32]]], list[list[NDArray[np.float32]]], list[Extrema]]: scale space [octave][scale], dogs [octave, scale-1], extremas
    """
    scale_space, deltas, sigmas = create_scale_space(img, sift_params)
    # show_array(scale_space, "Scale Space")
    dogs = create_dogs(scale_space, sift_params)
    normalized_dogs = [[cv.normalize(d, None, 0, 1, cv.NORM_MINMAX) for d in octave] for octave in dogs] # type: ignore
    # show_array(dogs, "Normalized DoGs")
    discrete_extremas = find_discrete_extremas(dogs, sift_params)
    # show_extremas(dogs, discrete_extremas, "Discrete Extremas")
    taylor_extremas = taylor_expansion(discrete_extremas,
                                       dogs,
                                       sift_params,
                                       deltas,
                                       sigmas)
    # show_extremas(dogs, taylor_extremas, "Taylor Extremas")
    filtered_extremas = filter_extremas(taylor_extremas,
                                        dogs,
                                        sift_params)
    
    gradients = gradient_2d(scale_space, sift_params)
    # show_extremas(dogs, filtered_extremas, "Filtered Extremas")
    key_points = assign_orientations(filtered_extremas,
                                     scale_space,
                                     sift_params,
                                     gradients,
                                     deltas,
                                     sigmas)
    # show_extremas(scale_space, key_points, "Key Points", 0, 2, MAX_SCALE-1, True)
    descriptor_key_points = create_descriptors(key_points,
                                               scale_space,
                                               sift_params,
                                               gradients,
                                               deltas,
                                               sigmas)
    # show_extremas(scale_space, descriptor_key_points, "Descriptor Key Points", 0, 2, MAX_SCALE-1, True, True)
    
    if(show_plots):
        show_array(scale_space, "Scale Space")
        show_array(dogs, "Normalized DoGs")
        show_extremas(dogs, discrete_extremas, "Discrete Extremas", sift_params.n_spo+1)
        show_extremas(dogs, taylor_extremas, "Taylor Extremas", sift_params.n_spo+1)
        show_extremas(dogs, filtered_extremas, "Filtered Extremas", sift_params.n_spo+1)
        show_extremas(scale_space, key_points, "Key Points", sift_params.n_spo+1, 0, True)
        show_extremas(scale_space, descriptor_key_points, "Descriptor Key Points", sift_params.n_spo+1, 0, True, True)
    
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
    sift_params = SIFT_Params()
    scale_space, normalized_dogs, key_points = detect_and_compute(gray, sift_params, show_plots=True)
    rotated_scale_space, rotated_normalized_dogs, roated_key_points = detect_and_compute(rotated, sift_params, show_plots=True)

    # Keypoint Matching
    
if __name__ == "__main__":
    main()
