import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import scipy

from SIFT_KeyPoint import KeyPoint
from SIFT_Params import SIFT_Params
from SIFT_Visualization import visualize_scale_space, visualize_keypoints
        


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
                if (alpha[0,0] > 0.6 and s+1 < len(dog_scales[extremum.o-1])-1):
                    s += 1
                elif(alpha[0,0] < -0.6 and s-1 > 0):
                    s -= 1
                if (alpha[1,0] > 0.6 and m+1 < current.shape[1]-1):
                    m += 1
                elif(alpha[1,0] < -0.6 and m-1 > 0):
                    m -= 1
                if (alpha[2,0] > 0.6 and n+1 < current.shape[0]-1):
                    n += 1
                elif(alpha[2,0] < -0.6 and n-1 > 0):
                    n -= 1
                # this is simplified from 'w+alphaT*g + 0.5*alphaT*H*alpha'
                # to 'w+0.5*alphaT*g' following the paper
                # pseudocode does not simplify here
                # omega represent the value of the DoG interpolated extremum
                omega = current[extremum.n, extremum.m] + 0.5*alpha.T*g_o_smn
                # calculate the current delta and sigma for the corresponding new location
                delta_oe = deltas[extremum.o-1]
                # sigma is calculated from the scale
                sigma = sigmas[extremum.o-1][s]* pow(sigmas[0][1]-sigmas[0][0], alpha[0,0])
                # and the keypoint coordinates
                x = delta_oe*(alpha[1,0]+m)
                y = delta_oe*(alpha[2,0]+n)
                # create new Extremum object with the corresponding values
                new_extremum = KeyPoint(extremum.o, s, m, n, sigma, x, y, omega)
                new_extremas.append(new_extremum)
                break
            # if the new location is valid, update the locations in that direction
            # at least one value will be adjust (as at least one is >0.6)
            if (alpha[0,0] > 0.6 and s+1 < len(dog_scales[extremum.o-1])-1):
                s += 1
            elif(alpha[0,0] < -0.6 and s-1 > 0):
                s -= 1
            if (alpha[1,0] > 0.6 and m+1 < current.shape[1]-1):
                m += 1
            elif(alpha[1,0] < -0.6 and m-1 > 0):
                m -= 1
            if (alpha[2,0] > 0.6 and n+1 < current.shape[0]-1):
                n += 1
            elif(alpha[2,0] < -0.6 and n-1 > 0):
                n -= 1
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

def assign_orientations(keypoints: list[KeyPoint], 
                        scale_space: list[list[NDArray[np.float32]]],
                        sift_params: SIFT_Params,
                        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                        deltas: list[float]) -> list[KeyPoint]:
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
        image = scale_space[keypoint.o-1][keypoint.s]
        key_x = keypoint.x / deltas[keypoint.o-1]
        key_y = keypoint.y / deltas[keypoint.o-1]
        key_sigma = keypoint.sigma / deltas[keypoint.o-1]
        border_limit = 3*sift_params.lambda_ori*key_sigma
        if(border_limit <= key_x and
           key_x <= image.shape[1]-border_limit and
           border_limit <= key_y and
           key_y <= image.shape[0]-border_limit):
            h_k = np.zeros(sift_params.n_bins)
            for m in range(max(0,int((key_x-border_limit+0.5))), min(image.shape[1]-1, int(key_x+border_limit+0.5))):
                for n in range(max(0,int(key_y-border_limit+0.5)), min(image.shape[0]-1, int(key_y+border_limit+0.5))):
                    sM = (m-key_x)/key_sigma
                    sN = (n-key_y)/key_sigma
                    # calculate gradient magnitude and angle
                    delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                    magnitude = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                    # calculate gaussian weight
                    weight = np.exp((-(sM*sM+sN*sN))/(2*(sift_params.lambda_ori**2)))
                    angle = np.arctan2(delta_m, delta_n)
                    # calculate histogram index
                    # histogram is 0-based, but calculation is 1-based
                    if(angle < 0):
                        angle += 2*np.pi
                    b_index = int(angle/(2*np.pi)*sift_params.n_bins+0.5)%sift_params.n_bins
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
                if(h_k[k] > prev_hist and h_k[k] > next_hist and h_k[k] >= sift_params.t*max_h_k):
                    theta_k = (2*np.pi*k)/sift_params.n_bins
                    angle = theta_k+(np.pi/sift_params.n_bins)*((prev_hist-next_hist)/(prev_hist-2*h_k[k]+next_hist))
                    keypoint.theta = angle
                    new_keypoints.append(keypoint)
    return new_keypoints

def create_descriptors(keypoints: list[KeyPoint],
                       scale_space: list[list[NDArray[np.float32]]],
                       sift_params: SIFT_Params,
                       gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                       deltas: list[float]) -> list[KeyPoint]:
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
        image = scale_space[keypoint.o-1][keypoint.s]
        key_x = keypoint.x / deltas[keypoint.o-1]
        key_y = keypoint.y / deltas[keypoint.o-1]
        key_sigma = keypoint.sigma / deltas[keypoint.o-1]
        relative_patch_size = sift_params.lambda_descr*key_sigma*(sift_params.n_hist+1)/sift_params.n_hist
        border_limit = np.sqrt(2)*relative_patch_size
        if(border_limit <= key_x and
           key_x <= image.shape[1]-border_limit and
           border_limit <= key_y and
           key_y <= image.shape[0]-border_limit):
            histograms = [[np.zeros(sift_params.n_ori) for _ in range(sift_params.n_hist)] for _ in range(sift_params.n_hist)]
            for m in range(max(0, int(key_x - border_limit + 0.5)), min(image.shape[1]-1, int(key_x + border_limit + 0.5))):
                for n in range(max(0, int(key_y - border_limit + 0.5)), min(image.shape[0]-1, int(key_y + border_limit + 0.5))):
                    x = m - key_x
                    y = n - key_y
                    x_vedge_mn = np.cos(keypoint.theta)*x-np.sin(keypoint.theta)*y
                    y_vedge_mn = np.sin(keypoint.theta)*x+np.cos(keypoint.theta)*y
                    
                    if(max(abs(x_vedge_mn),abs(y_vedge_mn)) < relative_patch_size):
                        delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                        theta_mn = (np.arctan2(delta_m, delta_n) - keypoint.theta)%(2*np.pi)
                        magnitude = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                        weight = np.exp(-(x*x+y*y)/(2*(sift_params.lambda_descr*key_sigma)**2))
                        
                        alpha = x/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                        beta = y/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                        gamma = theta_mn / (2*np.pi)*sift_params.n_ori
                        for i  in range(max(0, int(alpha), min(int(alpha)+2, sift_params.n_hist))):
                                for j in range(max(0, int(beta)), min(int(beta)+2, sift_params.n_hist)):
                                    k = (int(gamma)+sift_params.n_ori)%sift_params.n_ori
                                    histograms[i][j][k] += (1. - (gamma-int(gamma)))*(1. - abs(float(i-alpha)))*(1. - abs(float(j-beta)))*weight*magnitude
                                    k = (int(gamma)+1+sift_params.n_ori)%sift_params.n_ori
                                    histograms[i][j][k] += (1. - (gamma-int(gamma)))*(1. - abs(float(i-alpha)))*(1. - abs(float(j-beta)))*weight*magnitude
            f = np.array(histograms).flatten()
            f_norm = np.linalg.norm(f, 2)
            for l in range(0,f.shape[0]):
                f[l] = min(f[l], 0.2*f_norm)
            f_norm = np.linalg.norm(f, 2)
            for l in range(0, f.shape[0]):
                f[l] = min(np.floor(512*f[l]/f_norm), 255)
            keypoint.descriptor = f
            new_keypoints.append(keypoint)
    return new_keypoints
    
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
    # Image scaled down to lowest octave must have at least 12 pixel width and height
    if (img.shape[0] < 2**(sift_params.n_oct-1)*12 or img.shape[1] < 2**(sift_params.n_oct-1)*12):
        raise ValueError("Image is too small for the given number of octaves")
    
    scale_space, deltas, sigmas = create_scale_space(img, sift_params)
    dogs = create_dogs(scale_space, sift_params)
    normalized_dogs = [[cv.normalize(d, None, 0, 1, cv.NORM_MINMAX) for d in octave] for octave in dogs] # type: ignore
    discrete_extremas = find_discrete_extremas(dogs, sift_params)
    taylor_extremas = taylor_expansion(discrete_extremas,
                                       dogs,
                                       sift_params,
                                       deltas,
                                       sigmas)
    filtered_extremas = filter_extremas(taylor_extremas,
                                        dogs,
                                        sift_params)
    gradients = gradient_2d(scale_space, sift_params)
    key_points = assign_orientations(filtered_extremas,
                                     scale_space,
                                     sift_params,
                                     gradients,
                                     deltas)
    descriptor_key_points = create_descriptors(key_points,
                                               scale_space,
                                               sift_params,
                                               gradients,
                                               deltas)
    
    if(show_plots):
        visualize_scale_space(scale_space, "Scale Space")
        visualize_scale_space(dogs, "Normalized DoGs")
        visualize_keypoints(dogs, discrete_extremas, "Discrete Extremas", sift_params.n_spo+1)
        visualize_keypoints(dogs, taylor_extremas, "Taylor Extremas", sift_params.n_spo+1)
        visualize_keypoints(dogs, filtered_extremas, "Filtered Extremas", sift_params.n_spo+1)
        visualize_keypoints(scale_space, key_points, "Key Points", sift_params.n_spo+1, 0, True, deltas, sift_params, True)
        visualize_keypoints(scale_space, descriptor_key_points, "Descriptor Key Points", sift_params.n_spo+1, 0, True, deltas, sift_params, True)
    
    return scale_space, normalized_dogs, descriptor_key_points

def match_keypoints(keypoints_a: list[KeyPoint], keypoints_b: list[KeyPoint], sift_params: SIFT_Params) -> dict[KeyPoint, list[KeyPoint]]:
    """Matches two sets of keypoints and returns the matched keypoints of b to a

    Args:
        keypoints_a (list[KeyPoint]): list of keypoints of image a
        keypoints_b (list[KeyPoint]): list of keypoints of image b
        sift_params (SIFT_Params): the parameters for the SIFT algorithm

    Returns:
        dict[KeyPoint, list[KeyPoint]]: the matched keypoints of b to a
    """
    matches: dict[KeyPoint, list[KeyPoint]] = dict[KeyPoint, list[KeyPoint]]()
    for keypoint_a in keypoints_a:
        distances = []
        key_points = []
        # compute distances
        for keypoint_b in keypoints_b:
            distance = np.linalg.norm(keypoint_a.descriptor-keypoint_b.descriptor, 2)
            # if distance is smaller than threshold
            if(distance < sift_params.C_match_absolute):
                distances.append(distance)
                key_points.append(keypoint_b)
        if(len(distances) == 0):
            continue
        distances = np.array(distances)
        # find minimum distance
        min_f_b_index = np.argmin(distances)
        # and extract distance and keypoint
        min_f_b = distances[min_f_b_index]
        keypoint_b = key_points[min_f_b_index]
        # remove from distances
        distances = np.delete(distances, min_f_b_index)
        # and find second minimum distance
        min_2_f_b = np.min(distances)
        # if the ratio of the two distances is smaller than threshold
        if(min_f_b < sift_params.C_match_relative*min_2_f_b):
            # add to matches
            if(keypoint_a in matches):
                matches[keypoint_a].append(keypoint_b)
            else:
                matches[keypoint_a] = [keypoint_b]
    return matches

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
    scale_space, normalized_dogs, keypoints = detect_and_compute(gray, sift_params, show_plots=True)
    rotated_scale_space, rotated_normalized_dogs, rotated_keypoints = detect_and_compute(rotated, sift_params, show_plots=True)

    # Keypoint Matching
    
    matched_keypoints = match_keypoints(keypoints, rotated_keypoints, sift_params)
    print(matched_keypoints)
    
if __name__ == "__main__":
    main()
