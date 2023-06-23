from typing import Tuple
import cv2 as cv
import numpy as np
import scipy


# custom imports
from SIFT_KeyPoint import SIFT_KeyPoint # represents a keypoint or extremum found
from SIFT_Params import SIFT_Params # represents all hyperparameters for the algorithm
   
# this implementation is based on http://www.ipol.im/pub/art/2014/82/article.pdf and follows their C-Implementation
class SIFT_Algorithm:
    """Represents the SIFT algorithm
    """
    @staticmethod
    def create_scale_space(u_in: np.ndarray, sift_params: SIFT_Params) -> Tuple[list[list[np.ndarray]],list[float],list[list[float]]]:
        """Creates a scale space for a given image

        Args:
            u_in (np.ndarray): the image
            sift_Params (SIFT_Params): the sift parameters

        Returns:
            Tuple[list[list[np.ndarray]], list[float], list[list[float]]]: the scale space divide into octave - scales - images, the delta values and the sigma values
        """
        # Variable Initialization
        scale_space: list[list[np.ndarray]] = []
        deltas: list[float] = [sift_params.delta_min]
        sigmas: list[list[float]] = [[sift_params.sigma_min]]
        # scale image up with bilinear interpolation
        # delta_min is usually 0.5, therefore we scale up by 2
        u: np.ndarray = cv.resize(u_in, (0,0), fx=1.0 / sift_params.delta_min, fy=1.0 / sift_params.delta_min, interpolation=cv.INTER_LINEAR)
        # seed image is then blurred
        # the sigma value is taken as the difference between the last two sigma values and the current delta value
        # default values results in sqrt(0.8^2 - 0.5^2)/0.5 = 1.248
        sigma_extra: float = np.sqrt(sift_params.sigma_min**2 - sift_params.sigma_in**2)/sift_params.delta_min
        # the original code proposes a kernel size of 4*sigma_extra
        # but this ksize often results in even number kernels
        # we therefore let cv calculate the kernel size
        # ksize = (int(np.ceil(4*sigma_extra)),int(np.ceil(4*sigma_extra)))
        
        # this represents the first scale image of the first octave
        # in the paper the octave are counted from 1 to n_oct
        # and scales from 0 to n_spo+2
        v_1_0: np.ndarray = cv.GaussianBlur(u, (0,0), sigma_extra)
        first_octave: list[np.ndarray] = [v_1_0]
        
        # first octave calculation
        # delta_o equates to delta_min here
        # range(x,y) is [x,y), therefore we need to add 1 to y
        # go through all scales + 2 as described in the paper
        for s in range(1, sift_params.n_spo+3):
            # calculate the current sigma value
            # this is taken from the C-Implementation
            # rather than the paper
            sigma: float = sift_params.sigma_min*pow(2.0,s/sift_params.n_spo)
            # and calculate the difference between the last two sigma values and divide by the current delta value
            sigma_extra = np.sqrt(sigma**2 - sigmas[0][s-1]**2)/sift_params.delta_min
            # save the sigma value
            sigmas[0].append(sigma)
            # and append the blurred image
            first_octave.append(cv.GaussianBlur(first_octave[s-1], (0,0), sigma_extra))
        scale_space.append(first_octave)
        
        # for every other octave excluding the first
        for o in range(1, sift_params.n_oct):
            # delta_o is the last delta doubled
            delta_o: float = deltas[o-1]*2
            deltas.append(delta_o)
            # seed image is antepenultimate image of previous octave (previous of the previous)
            # there are n_spo+2 images per octave with index 0-n_spo+1
            # therefore the antepenultimate image is at index n_spo-1
            seed: np.ndarray = scale_space[o-1][sift_params.n_spo-1]
            # save the sigma from the seed image
            sigmas.append([sigmas[o-1][sift_params.n_spo-1]])
            # scale image down with bilinear interpolation
            # this is generally delta_prev/delta_o
            # but as we just double delta_prev, to get delta_o
            # we take 0.5 as the scaling factor here
            # cv.INTER_LINEAR is the binary interpolation method
            seed = cv.resize(seed, (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
            current_octave: list[np.ndarray] = [seed]
            # for every scale level
            for s in range(1, sift_params.n_spo+3):
                # calculate sigma
                # this is the original code from the paper
                sigma = delta_o/sift_params.delta_min*sift_params.sigma_min*pow(2.0, s/sift_params.n_spo)
                # and the corresponding sigma_extra
                sigma_extra = np.sqrt(sigma**2 - sigmas[o][s-1]**2)/delta_o
                # append the sigma value
                sigmas[o].append(sigma)
                # and apply blur to previous scale image
                current_octave.append(cv.GaussianBlur(current_octave[s-1], (0,0), sigma_extra))
            # and append octave
            scale_space.append(current_octave)
        return scale_space, deltas, sigmas

    @staticmethod
    def create_dogs(scale_space: list[list[np.ndarray]], sift_params: SIFT_Params) -> list[list[np.ndarray]]:
        """Creates the difference of gaussians for a given scale space

        Args:
            scale_space (list[list[np.ndarray]]): the scale space
            sift_params (SIFT_Params): the sift parameters

        Returns:
            list[list[np.ndarray]]: the difference of gaussians
        """
        dogs: list[list[np.ndarray]] = []
        # for each octave
        for o in range(sift_params.n_oct):
            octave: list[np.ndarray] = scale_space[o]
            dog_row: list[np.ndarray] = []
            # for each scale level (excluding last image)
            for s in range(sift_params.n_spo+2):
                # calculate difference of gaussians
                # cv.subtract should subtract elementwise here
                dog_row.append(cv.subtract(octave[s+1], octave[s]))
            dogs.append(dog_row)
        return dogs

    @staticmethod
    def find_discrete_extremas(dogs: list[list[np.ndarray]], sift_params: SIFT_Params, sigmas: list[list[float]], deltas: list[float]) -> list[SIFT_KeyPoint]:
        """Finds the discrete extremas for a given difference of gaussians

        Args:
            dogs (list[list[np.ndarray]]): the difference of gaussians
            sift_params (SIFT_Params): the sift parameters (used for n_oct and n_spo)

        Returns:
            list[KeyPoint]: the discrete extremas
        """
        extremas: list[SIFT_KeyPoint] = []
        # for each octave in dogs
        for o in range(sift_params.n_oct):
            # this code generally takes the longest
            # we print a little progress indicator
            print(f"Extrema Calculation: Octave {o}")
            octave: list[np.ndarray] = dogs[o]
            # for each dog image in the octave excluding first and last image
            for s in range(1, sift_params.n_spo+1):
                # the current dog image
                current: np.ndarray = octave[s]
                # the one before
                before: np.ndarray = octave[s-1]
                # the one after
                after: np.ndarray = octave[s+1]
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
                            extremas.append(SIFT_KeyPoint(o,s,m,n,sigmas[o][s], deltas[o]*m, deltas[o]*n))
        return extremas

    @staticmethod
    def taylor_expansion(extremas: list[SIFT_KeyPoint],
                        dog_scales: list[list[np.ndarray]],
                        sift_params: SIFT_Params,
                        deltas: list[float],
                        sigmas: list[list[float]]) -> list[SIFT_KeyPoint]:
        """Finetunes locations of extrema using taylor expansion

        Args:
            extremas (list[KeyPoint]): The extremas to finetune
            dog_scales (list[list[np.ndarray]]): The difference of gaussians images to finetune with
            sift_params (SIFT_Params): The sift parameters
            deltas (list[float]): The deltas for each octave
            sigmas (list[list[float]]): The sigmas for each octave and scale level
        Returns:
            list[KeyPoint]: The new Extremum. Newly created KeyPoint Objects.
        """
        new_extremas: list[SIFT_KeyPoint] = []
        for extremum in extremas:
            # discard low contrast candidate keypoints
            # this step is done separately in the C-Implementation
            # but can be included here for "slightly" better performance
            if(abs(dog_scales[extremum.o][extremum.s][extremum.n, extremum.m]) < 0.8 * sift_params.C_DoG):
                continue
            # 0-based index of the current extremum
            s: int = extremum.s # scale level
            m: int = extremum.m # x-coordinate
            n: int = extremum.n # y-coordinate
            # for each adjustment
            # will break if new location is found
            # will be adjusted maximum 5 times
            for _ in range(5):
                # get the current dog image
                # s is initialized from dog images
                # but also represent the scale level
                current: np.ndarray = dog_scales[extremum.o][s]
                previous: np.ndarray = dog_scales[extremum.o][s-1]
                next_scale: np.ndarray = dog_scales[extremum.o][s+1]
                
                # called $\bar{g}^o_{s,m,n}$ in the paper
                # represent the first derivative  in a finite difference scheme
                # is Transposed, as we calculate [a,b,c] values, but want [[a],[b],[c]]
                g_o_smn = np.matrix([[next_scale[n, m] - previous[n, m], current[n, m+1] - current[n, m-1], current[n+1, m]- current[n-1, m]]]).T
                
                # calcuation of hessian matrix
                # note that s is the first dimension, then x, then y
                h11: float = (next_scale[n, m] - previous[n, m] - 2*current[n, m])
                h22: float = (current[n, m+1] + current[n, m-1] - 2*current[n, m])
                h33: float = (current[n+1, m]- current[n-1, m] - 2*current[n, m])
                
                # h12, h13 and h23 are reused for h21, h31 and h32
                # as they are the same value
                h12: float = (next_scale[n, m+1] - next_scale[n, m-1]- previous[n, m+1] - previous[n, m-1])/4
                h13: float = (next_scale[n+1, m] - next_scale[n-1, m]- previous[n+1, m] - previous[n-1, m])/4
                h23: float = (current[n+1, m+1] - current[n+1, m-1]-current[n-1, m+1] - current[n-1, m-1])/4
                
                hessian = np.matrix([[h11, h12, h13],
                                    [h12, h22, h23], 
                                    [h13, h23, h33]])
                # inverse of a matrix with det = 0 is not possible
                # therefore we break here
                # this is just safety, should not happen
                if(np.linalg.det(hessian) == 0):
                    break
                # calculate offset, this will be of shape (1,3) ([[a],[b],[c]])
                alpha = -np.linalg.inv(hessian) * g_o_smn
                # if every value is below the drop off of 0.6
                # we found the new location
                if(np.max(np.abs(alpha)) < 0.6):
                    # this is simplified from 'w+alphaT*g + 0.5*alphaT*H*alpha'
                    # to 'w+0.5*alphaT*g' following the paper
                    # pseudocode in the paper does not simplify here
                    # omega represent the value of the DoG interpolated extremum
                    omega = current[extremum.n, extremum.m] + 0.5*(g_o_smn[0,0]*alpha[0,0]+g_o_smn[1,0]*alpha[1,0]+g_o_smn[2,0]*alpha[2,0])
                    # get the current delta and sigma for the corresponding new location
                    delta_oe = deltas[extremum.o]
                    # sigma is calculated from the scale
                    sigma = sigmas[extremum.o][s]* pow(sigmas[0][1]-sigmas[0][0], alpha[0,0])
                    # and the keypoint coordinates
                    x = delta_oe*(alpha[1,0]+m)
                    y = delta_oe*(alpha[2,0]+n)
                    # create new Extremum object with the corresponding values
                    new_extremum = SIFT_KeyPoint(extremum.o, s, m, n, sigma, x, y, omega)
                    new_extremas.append(new_extremum)
                    break
                # update coordinates by +1 or -1 if the abs(alpha) is > 0.6
                # but borders are not reached
                # we could optionally exclude extremum that are close to the border
                # but those are also excluded later on aswell
                if (alpha[0,0] > 0.6 and s+1 < len(dog_scales[extremum.o])-1):
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

    @staticmethod
    def filter_extremas(extremas: list[SIFT_KeyPoint], dogs: list[list[np.ndarray]], sift_params: SIFT_Params) -> list[SIFT_KeyPoint]:
        """Filters Extrema based on contrast and curvature

        Args:
            extremas (list[KeyPoint]): The extrema to filter
            dogs (list[list[np.ndarray]]): The dog values to calculate curvature from
            sift_params (SIFT_Params): The sift parameters

        Returns:
            list[KeyPoint]: Filtered Extremum. Returns same objects from input.
        """
        filtered_extremas: list[SIFT_KeyPoint] = []
        for extremum in extremas:
            # current location of the extremum
            s: int = extremum.s # scale index
            m: int = extremum.m # x-coordinate
            n: int = extremum.n # y-coordinate
            # current dog image
            current: np.ndarray = dogs[extremum.o][s]
            
            # contrast drop off from the calculate omega value of taylor expansion
            if(abs(extremum.omega) < sift_params.C_DoG):
                continue
            # filter off extremas at the border
            if(m < 1 or m > current.shape[1]-2 or n < 1 or n > current.shape[0]-2):
                continue
            
            # 2d-Hessian matrix over x and y
            h11: float = (current[n, m+1] - current[n, m-1]) / 2
            h22: float = (current[n+1, m]- current[n-1, m]) / 2
            # h12 will be resued for h21
            h12 = (current[n+1, m+1] - current[n+1, m-1]-current[n-1, m+1] - current[n-1, m-1])/4
            
            hessian = np.matrix([[h11, h12],
                                [h12, h22]])
            
            trace = np.trace(hessian)
            determinant = np.linalg.det(hessian)
            # if we divide by 0, extremum is not valid
            if(determinant == 0):
                continue
            edgeness: float = (trace*trace)/determinant
            
            # curvature drop off
            if(abs(edgeness) >= ((sift_params.C_edge+1)**2)/sift_params.C_edge):
                continue
            
            filtered_extremas.append(extremum)
        return filtered_extremas

    @staticmethod
    def gradient_2d(scale_space: list[list[np.ndarray]], sift_params: SIFT_Params) -> dict[Tuple[int, int, int, int], Tuple[float, float]]:
        """Calculate the 2d gradient for each pixel in the scale space

        Args:
            scale_space (list[list[np.ndarray]]): the scale space to calculate the gradient from
            sift_params (SIFT_Params): the sift parameters

        Returns:
            dict[Tuple[int, int, int, int], Tuple[float, float]]: Dictionary containing each x and y gradients. 
            Key is (o, s, m, n) and value is (grad_x, grad_y)
        """
        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
        for o in range(sift_params.n_oct):
            # include all images from 0 to n_spo+2
            for s in range(sift_params.n_spo+3):
                # get current image
                v_o_s: np.ndarray = scale_space[o][s]
                # for each pixel in the image
                for m in range(0, v_o_s.shape[1]):
                    for n in range(0, v_o_s.shape[0]):
                        # if pixel is at the border, we use the difference to the other side
                        if (m == 0):
                            delta_m = (v_o_s[n, v_o_s.shape[1]-1]-v_o_s[n, 1]) / 2
                        elif (m  == v_o_s.shape[1]-1):
                            delta_m = (v_o_s[n, 0]-v_o_s[n, v_o_s.shape[1]-2]) / 2
                        # otherwise difference to the left and right pixel
                        else:
                            delta_m = (v_o_s[n, m+1]-v_o_s[n, m-1])/2
                        # same for y
                        if(n == 0):
                            delta_n = (v_o_s[v_o_s.shape[0]-1, m]-v_o_s[1, m]) / 2
                        elif(n == v_o_s.shape[0]-1):
                            delta_n = (v_o_s[0, m]-v_o_s[v_o_s.shape[0]-2, m]) / 2
                        else:
                            delta_n = (v_o_s[n+1, m]-v_o_s[n-1, m])/2
                        gradients[(o, s, m, n)] = (delta_m, delta_n)
        return gradients

    @staticmethod
    def _float_modulo(val: float, mod: float) -> float:
        """Performs modulo operation with float modulo

        Args:
            val (float): the value
            mod (float): the modulo

        Returns:
            float: val % mod with float modulo
        """
        z: float = val
        n: int = 0
        # if value is negative
        if (z < 0):
            # get number of mods fit into value + 1 as we want to make it positive
            n = int((-z)/mod)+1
            # and add them
            z += n*mod
        # get number of mods fit into value
        n = int(z/mod)
        # and subtract them
        z -= n*mod
        return z

    @staticmethod
    def assign_orientations(keypoints: list[SIFT_KeyPoint], 
                            scale_space: list[list[np.ndarray]],
                            sift_params: SIFT_Params,
                            gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                            deltas: list[float]) -> list[SIFT_KeyPoint]:
        """Assign orientation angle and magnitude to each Keypoint.

        Args:
            keypoints (list[KeyPoint]): List of keypoints to assign orientation to
            scale_space (list[list[np.ndarray]]): The scale space to calculate from
            sift_params (SIFT_Params): The sift parameters
            gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
            deltas (list[float]): list of deltas for each octave

        Returns:
            list[KeyPoint]: List of keypoints with assigned orientation. KeyPoint are new objects.
        """
        new_keypoints: list[SIFT_KeyPoint] = []
        for keypoint in keypoints:
            # current image
            image = scale_space[keypoint.o][keypoint.s]
            # as x and y are calculate in taylor expansion with m*delta or n*delta, we need to divide by delta
            # this follows the C-implementation
            key_x = keypoint.x / deltas[keypoint.o]
            key_y = keypoint.y / deltas[keypoint.o]
            key_sigma = keypoint.sigma / deltas[keypoint.o]
            # border limit is half the window size
            # this depends on the sigma of the keypoint
            border_limit = 3*sift_params.lambda_ori*key_sigma
            # if keypoint is not at the border
            if(border_limit <= key_x and
            key_x <= image.shape[1]-border_limit and
            border_limit <= key_y and
            key_y <= image.shape[0]-border_limit):
                # initialize histogram
                h_k: np.ndarray = np.zeros(sift_params.n_bins)
                # for each pixel in the window
                # this follows the C-implementation
                # and does not represent the paper pseudo code
                for m in range(max(0,int((key_x-border_limit+0.5))), min(image.shape[1]-1, int(key_x+border_limit+0.5))):
                    for n in range(max(0,int(key_y-border_limit+0.5)), min(image.shape[0]-1, int(key_y+border_limit+0.5))):
                        # calculate the normalized positions
                        sM = (m-key_x)/key_sigma
                        sN = (n-key_y)/key_sigma
                        # get gradients
                        delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                        # calculate magnitude
                        magnitude: float = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                        # calculate gaussian weight
                        weight: float = np.exp((-(sM*sM+sN*sN))/(2*(sift_params.lambda_ori**2)))
                        # and angle
                        angle = SIFT_Algorithm._float_modulo(np.arctan2(delta_m, delta_n), 2*np.pi)
                        # calculate histogram index
                        if(angle < 0):
                            angle += 2*np.pi
                        b_index = int(angle/(2*np.pi)*sift_params.n_bins+0.5)%sift_params.n_bins
                        # add weight to histogram
                        h_k[b_index] += weight*magnitude
                # smooth histogram
                for _ in range(6):
                    # mode wrap represents a circular convolution
                    h_k = scipy.ndimage.convolve1d(h_k, weights=[1/3,1/3,1/3], mode='wrap')
                # maximum of the histogram
                max_h_k = np.max(h_k)
                # for each histogram bin
                for k in range(sift_params.n_bins):
                    # get previous and next histogram bin
                    prev_hist: np.float32 = h_k[(k-1+sift_params.n_bins)%sift_params.n_bins]
                    next_hist: np.float32 = h_k[(k+1)%sift_params.n_bins]
                    # if current bin is above threshold and is a local maximum
                    if(h_k[k] > prev_hist and h_k[k] > next_hist and h_k[k] >= sift_params.t*max_h_k):
                        # calculate offset
                        offset: float = (prev_hist - next_hist)/(2*(prev_hist+next_hist-2*h_k[k]))
                        # and new angle
                        angle = (k+offset+0.5)*2*np.pi/sift_params.n_bins
                        if(angle > 2*np.pi):
                            angle -= 2*np.pi
                        new_keypoint = SIFT_KeyPoint(keypoint.o,
                                                keypoint.s,
                                                keypoint.m,
                                                keypoint.n,
                                                keypoint.sigma,
                                                keypoint.x,
                                                keypoint.y,
                                                keypoint.omega,
                                                angle,
                                                keypoint.magnitude)
                        new_keypoints.append(new_keypoint)
        return new_keypoints

    @staticmethod
    def create_descriptors(keypoints: list[SIFT_KeyPoint],
                        scale_space: list[list[np.ndarray]],
                        sift_params: SIFT_Params,
                        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]],
                        deltas: list[float]) -> list[SIFT_KeyPoint]:
        """Creates Key Descriptors for each Keypoint.

        Args:
            extremas (list[KeyPoint]): The keypoints to create descriptors for
            scale_space (list[list[np.ndarray]]): The scalespace to calculate from
            sift_params (SIFT_Params): The sift parameters
            gradients (dict[Tuple[int, int, int, int], Tuple[float, float]]): The gradients for each pixel
            deltas (list[float]): list of deltas for each octave

        Returns:
            list[KeyPoint]: The keypoints with descriptors. KeyPoint are same objects.
        """
        new_keypoints: list[SIFT_KeyPoint] = []
        for keypoint in keypoints:
            # current image
            image = scale_space[keypoint.o][keypoint.s]
            # current location
            key_x = keypoint.x / deltas[keypoint.o]
            key_y = keypoint.y / deltas[keypoint.o]
            key_sigma = keypoint.sigma / deltas[keypoint.o]
            # relative patch size
            relative_patch_size = (1+1/sift_params.n_hist)*sift_params.lambda_descr*key_sigma
            # and the actual border limit
            border_limit = np.sqrt(2)*relative_patch_size
            if(border_limit <= key_x and
            key_x <= image.shape[1]-border_limit and
            border_limit <= key_y and
            key_y <= image.shape[0]-border_limit):
                # initialize histograms
                histograms: list[list[np.ndarray]] = [[np.zeros(sift_params.n_ori) for _ in range(sift_params.n_hist)] for _ in range(sift_params.n_hist)]
                # for each pixel in patch
                # this follows C-implementation
                # and deviates from the paper pseudo code
                for m in range(max(0, int(key_x - border_limit + 0.5)), min(image.shape[1]-1, int(key_x + border_limit + 0.5))):
                    for n in range(max(0, int(key_y - border_limit + 0.5)), min(image.shape[0]-1, int(key_y + border_limit + 0.5))):
                        # normalized positions by angle of keypoint
                        x_vedge_mn = np.cos(-keypoint.theta)*(m - key_x)-np.sin(-keypoint.theta)*(n - key_y)
                        y_vedge_mn = np.sin(-keypoint.theta)*(m - key_x)+np.cos(-keypoint.theta)*(n - key_y)
                        # if pixel is in patch
                        if(max(abs(x_vedge_mn),abs(y_vedge_mn)) < relative_patch_size):
                            # get gradient
                            delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                            # calculate new angle, subtract the keypoint angle
                            theta_mn =  SIFT_Algorithm._float_modulo((np.arctan2(delta_m, delta_n) - keypoint.theta), 2*np.pi)
                            magnitude: float = np.sqrt(delta_m*delta_m+delta_n*delta_n)
                            weight: float = np.exp(-(x_vedge_mn*x_vedge_mn+y_vedge_mn*y_vedge_mn)/(2*(sift_params.lambda_descr*key_sigma)**2))
                            # x and y histogram coordinates
                            alpha = x_vedge_mn/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                            beta = y_vedge_mn/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                            # and bin coordinate
                            gamma = theta_mn / (2*np.pi)*sift_params.n_ori
                            # for each histogram
                            for i  in range(max(0, int(alpha), min(int(alpha)+1, sift_params.n_hist-1)+1)):
                                    for j in range(max(0, int(beta)), min(int(beta)+1, sift_params.n_hist-1)+1):
                                        # get bin index to the left
                                        k = (int(gamma)+sift_params.n_ori)%sift_params.n_ori
                                        # add weight
                                        temp = (1. - (gamma-np.floor(gamma)))
                                        temp *= (1. - abs(float(i-alpha)))
                                        temp *= (1. - abs(float(j-beta)))
                                        temp *= weight*magnitude
                                        histograms[i][j][k] += temp
                                        # get bin to the right
                                        k = (int(gamma)+1+sift_params.n_ori)%sift_params.n_ori
                                        # add weight
                                        temp = (1. - (gamma-np.floor(gamma)))
                                        temp *= (1. - abs(float(i-alpha)))
                                        temp *= (1. - abs(float(j-beta)))
                                        temp *= weight*magnitude
                                        histograms[i][j][k] += temp
                # create descriptor vector
                f = np.array(histograms).flatten()
                # this is L2 normalization (sqrt(x^2+y^2+...))
                f_norm = np.linalg.norm(f, 2)
                # cap at 0.2*norm
                for l in range(0,f.shape[0]):
                    f[l] = min(f[l], 0.2*f_norm)
                # recalcualte norm
                f_norm = np.linalg.norm(f, 2)
                # quantize to 0-255
                for l in range(0, f.shape[0]):
                    f[l] = min(np.floor(512*f[l]/f_norm), 255)
                # set descriptor vector
                keypoint.descriptor = f
                new_keypoints.append(keypoint)
        return new_keypoints
    
    @staticmethod
    def match_keypoints(keypoints_a: list[SIFT_KeyPoint], keypoints_b: list[SIFT_KeyPoint], sift_params: SIFT_Params) -> dict[SIFT_KeyPoint, list[SIFT_KeyPoint]]:
        """Matches two sets of keypoints and returns the matched keypoints of b to a

        Args:
            keypoints_a (list[KeyPoint]): list of keypoints of image a
            keypoints_b (list[KeyPoint]): list of keypoints of image b
            sift_params (SIFT_Params): the parameters for the SIFT algorithm

        Returns:
            dict[KeyPoint, list[KeyPoint]]: the matched keypoints of b to a
        """
        matches: dict[SIFT_KeyPoint, list[SIFT_KeyPoint]] = dict[SIFT_KeyPoint, list[SIFT_KeyPoint]]()
        for keypoint_a in keypoints_a:
            # calculate distances to all keypoints in b
            distances: list[float] = []
            # corresponding keypoints to distances
            key_points: list[SIFT_KeyPoint] = []
            # compute distances
            for keypoint_b in keypoints_b:
                # L2 norm as distance
                distance = np.linalg.norm(keypoint_a.descriptor-keypoint_b.descriptor, 2)
                # if distance is smaller than threshold
                if(distance < sift_params.C_match_absolute):
                    distances.append(distance)
                    key_points.append(keypoint_b)
            # if only one or no keypoint is close enough
            # discard the keypoint
            if(len(distances) < 2):
                continue
            # convert to numpy array
            distances = np.array(distances)
            # find minimum distance index
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