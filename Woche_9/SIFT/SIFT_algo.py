O=pow
Q=float
L=max
J=1.
I=min
H=abs
C=int
D=range
from typing import Tuple
import cv2 as E,numpy as A,scipy
from SIFT_KeyPoint import SIFT_KeyPoint
from SIFT_Params import SIFT_Params

class SIFT_Algorithm:
    'Represents the SIFT algorithm\n    '
    # Methods are obfuscated by purpose as they are meant to be used only for comparison
    @staticmethod
    def create_scale_space(u_in,sift_params):
        'Creates a scale space for a given image\n\n        Args:\n            u_in (np.ndarray): the image\n            sift_Params (SIFT_Params): the sift parameters\n\n        Returns:\n            Tuple[list[list[np.ndarray]], list[float], list[list[float]]]: the scale space divide into octave - scales - images, the delta values and the sigma values\n        ';B=sift_params;K=[];L=[B.delta_min];C=[[B.sigma_min]];R=E.resize(u_in,(0,0),fx=J/B.delta_min,fy=J/B.delta_min,interpolation=E.INTER_LINEAR);G=A.sqrt(B.sigma_min**2-B.sigma_in**2)/B.delta_min;S=E.GaussianBlur(R,(0,0),G);M=[S]
        for F in D(1,B.n_spo+3):H=B.sigma_min*O(2.,F/B.n_spo);G=A.sqrt(H**2-C[0][F-1]**2)/B.delta_min;C[0].append(H);M.append(E.GaussianBlur(M[F-1],(0,0),G))
        K.append(M)
        for I in D(1,B.n_oct):
            N=L[I-1]*2;L.append(N);P=K[I-1][B.n_spo-1];C.append([C[I-1][B.n_spo-1]]);P=E.resize(P,(0,0),fx=.5,fy=.5,interpolation=E.INTER_LINEAR);Q=[P]
            for F in D(1,B.n_spo+3):H=N/B.delta_min*B.sigma_min*O(2.,F/B.n_spo);G=A.sqrt(H**2-C[I][F-1]**2)/N;C[I].append(H);Q.append(E.GaussianBlur(Q[F-1],(0,0),G))
            K.append(Q)
        return K,L,C
    @staticmethod
    def create_dogs(scale_space,sift_params):
        'Creates the difference of gaussians for a given scale space\n\n        Args:\n            scale_space (list[list[np.ndarray]]): the scale space\n            sift_params (SIFT_Params): the sift parameters\n\n        Returns:\n            list[list[np.ndarray]]: the difference of gaussians\n        ';A=sift_params;B=[]
        for H in D(0,A.n_oct):
            C=scale_space[H];F=[]
            for G in D(A.n_spo+2):F.append(E.subtract(C[G+1],C[G]))
            B.append(F)
        return B
    @staticmethod
    def find_discrete_extremas(dogs,sift_params,sigmas,deltas):
        'Finds the discrete extremas for a given difference of gaussians\n\n        Args:\n            dogs (list[list[np.ndarray]]): the difference of gaussians\n            sift_params (SIFT_Params): the sift parameters (used for n_oct and n_spo)\n\n        Returns:\n            list[KeyPoint]: the discrete extremas\n        ';K=deltas;J=sift_params;L=[]
        for F in D(0,J.n_oct):
            print(f"Extrema Calculation: Octave {F}");I=dogs[F]
            for G in D(1,J.n_spo+1):
                H=I[G];N=I[G-1];O=I[G+1]
                for B in D(1,H.shape[0]-1):
                    for C in D(1,H.shape[1]-1):
                        E=H[B-1:B+2,C-1:C+2].flatten();E=A.delete(E,4);E=A.append(E,N[B-1:B+2,C-1:C+2].flatten());E=A.append(E,O[B-1:B+2,C-1:C+2].flatten())
                        if A.max(E)<H[B,C]or A.min(E)>H[B,C]:L.append(SIFT_KeyPoint(F,G,C,B,sigmas[F][G],K[F]*C,K[F]*B))
        return L
    
    
    # non-minized function from here on
    @staticmethod
    def taylor_expansion(extremas: list[SIFT_KeyPoint],
                        dog_scales: list[list[A.ndarray]],
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
                current: A.ndarray = dog_scales[extremum.o][s]
                previous: A.ndarray = dog_scales[extremum.o][s-1]
                next_scale: A.ndarray = dog_scales[extremum.o][s+1]

                # called $\bar{g}^o_{s,m,n}$ in the paper
                # represent the first derivative  in a finite difference scheme
                # is Transposed, as we calculate [a,b,c] values, but want [[a],[b],[c]]
                g_o_smn = A.matrix([[next_scale[n, m] - previous[n, m], current[n, m+1] - current[n, m-1], current[n+1, m]- current[n-1, m]]]).T

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

                hessian = A.matrix([[h11, h12, h13],
                                    [h12, h22, h23], 
                                    [h13, h23, h33]])
                # inverse of a matrix with det = 0 is not possible
                # therefore we break here
                # this is just safety, should not happen
                if(A.linalg.det(hessian) == 0):
                    break
                # calculate offset, this will be of shape (1,3) ([[a],[b],[c]])
                alpha = -A.linalg.inv(hessian) * g_o_smn
                # the every value is below the drop off of 0.6
                # we found the new location
                if(A.max(A.abs(alpha)) < 0.6):
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
    def filter_extremas(extremas: list[SIFT_KeyPoint], dogs: list[list[A.ndarray]], sift_params: SIFT_Params) -> list[SIFT_KeyPoint]:
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
            current: A.ndarray = dogs[extremum.o][s]

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

            hessian = A.matrix([[h11, h12],
                                [h12, h22]])

            trace = A.trace(hessian)
            determinant = A.linalg.det(hessian)
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
    def gradient_2d(scale_space: list[list[A.ndarray]], sift_params: SIFT_Params) -> dict[Tuple[int, int, int, int], Tuple[float, float]]:
        """Calculate the 2d gradient for each pixel in the scale space

        Args:
            scale_space (list[list[np.ndarray]]): the scale space to calculate the gradient from
            sift_params (SIFT_Params): the sift parameters

        Returns:
            dict[Tuple[int, int, int, int], Tuple[float, float]]: Dictionary containing each x and y gradients. 
            Key is (o, s, m, n) and value is (grad_x, grad_y)
        """
        gradients: dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
        for o in range(0, sift_params.n_oct):
            # include all images from 0 to n_spo+2
            for s in range(0, sift_params.n_spo+3):
                # get current image
                v_o_s: A.ndarray = scale_space[o][s]
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
                            scale_space: list[list[A.ndarray]],
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
            list[KeyPoint]: List of keypoints with assigned orientation. KeyPoint are same objects.
        """
        new_keypoints: list[SIFT_KeyPoint] = []
        for keypoint in keypoints:
            # current image
            image = scale_space[keypoint.o][keypoint.s]
            # as x and y are calculate in taylor expansion with m*delta or n*delta, we need to divide by delta
            # this follow the C-implementation
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
                h_k: A.ndarray = A.zeros(sift_params.n_bins)
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
                        magnitude: float = A.sqrt(delta_m*delta_m+delta_n*delta_n)
                        # calculate gaussian weight
                        weight: float = A.exp((-(sM*sM+sN*sN))/(2*(sift_params.lambda_ori**2)))
                        # and angle
                        angle = SIFT_Algorithm._float_modulo(A.arctan2(delta_m, delta_n), 2*A.pi)
                        # calculate histogram index
                        if(angle < 0):
                            angle += 2*A.pi
                        b_index = int(angle/(2*A.pi)*sift_params.n_bins+0.5)%sift_params.n_bins
                        # add weight to histogram
                        h_k[b_index] += weight*magnitude
                # smooth histogram
                for _ in range(6):
                    # mode wrap represents a circular convolution
                    h_k = scipy.ndimage.convolve1d(h_k, weights=[1/3,1/3,1/3], mode='wrap')
                # maximum of the histogram
                max_h_k = A.max(h_k)
                # for each histogram bin
                for k in range(sift_params.n_bins):
                    # get previous and next histogram bin
                    prev_hist: A.float32 = h_k[(k-1+sift_params.n_bins)%sift_params.n_bins]
                    next_hist: A.float32 = h_k[(k+1)%sift_params.n_bins]
                    # if current bin is above threshold and is a local maximum
                    if(h_k[k] > prev_hist and h_k[k] > next_hist and h_k[k] >= sift_params.t*max_h_k):
                        # calculate offset
                        offset: float = (prev_hist - next_hist)/(2*(prev_hist+next_hist-2*h_k[k]))
                        # and new angle
                        angle = (k+offset+0.5)*2*A.pi/sift_params.n_bins
                        if(angle > 2*A.pi):
                            angle -= 2*A.pi
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
                        scale_space: list[list[A.ndarray]],
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
            border_limit = A.sqrt(2)*relative_patch_size
            if(border_limit <= key_x and
            key_x <= image.shape[1]-border_limit and
            border_limit <= key_y and
            key_y <= image.shape[0]-border_limit):
                # initialize histograms
                histograms: list[list[A.ndarray]] = [[A.zeros(sift_params.n_ori) for _ in range(sift_params.n_hist)] for _ in range(sift_params.n_hist)]
                # for each pixel in patch
                # this follows C-implementation
                # and deviates from the paper pseudo code
                for m in range(max(0, int(key_x - border_limit + 0.5)), min(image.shape[1]-1, int(key_x + border_limit + 0.5))):
                    for n in range(max(0, int(key_y - border_limit + 0.5)), min(image.shape[0]-1, int(key_y + border_limit + 0.5))):
                        # normalized positions by angle of keypoint
                        x_vedge_mn = A.cos(-keypoint.theta)*(m - key_x)-A.sin(-keypoint.theta)*(n - key_y)
                        y_vedge_mn = A.sin(-keypoint.theta)*(m - key_x)+A.cos(-keypoint.theta)*(n - key_y)
                        # if pixel is in patch
                        if(max(abs(x_vedge_mn),abs(y_vedge_mn)) < relative_patch_size):
                            # get gradient
                            delta_m, delta_n = gradients[(keypoint.o, keypoint.s, m, n)]
                            # calculate new angle, subtract the keypoint angle
                            theta_mn =  SIFT_Algorithm._float_modulo((A.arctan2(delta_m, delta_n) - keypoint.theta), 2*A.pi)
                            magnitude: float = A.sqrt(delta_m*delta_m+delta_n*delta_n)
                            weight: float = A.exp(-(x_vedge_mn*x_vedge_mn+y_vedge_mn*y_vedge_mn)/(2*(sift_params.lambda_descr*key_sigma)**2))
                            # x and y histogram coordinates
                            alpha = x_vedge_mn/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                            beta = y_vedge_mn/(2*sift_params.lambda_descr*key_sigma/sift_params.n_hist) + (sift_params.n_hist-1)/2
                            # and bin coordinate
                            gamma = theta_mn / (2*A.pi)*sift_params.n_ori
                            # for each histogram
                            for i  in range(max(0, int(alpha), min(int(alpha)+1, sift_params.n_hist-1)+1)):
                                    for j in range(max(0, int(beta)), min(int(beta)+1, sift_params.n_hist-1)+1):
                                        # get bin index to the left
                                        k = (int(gamma)+sift_params.n_ori)%sift_params.n_ori
                                        # add weight
                                        temp = (1. - (gamma-A.floor(gamma)))
                                        temp *= (1. - abs(float(i-alpha)))
                                        temp *= (1. - abs(float(j-beta)))
                                        temp *= weight*magnitude
                                        histograms[i][j][k] += temp
                                        # get bin to the right
                                        k = (int(gamma)+1+sift_params.n_ori)%sift_params.n_ori
                                        # add weight
                                        temp = (1. - (gamma-A.floor(gamma)))
                                        temp *= (1. - abs(float(i-alpha)))
                                        temp *= (1. - abs(float(j-beta)))
                                        temp *= weight*magnitude
                                        histograms[i][j][k] += temp
                # create descriptor vector
                f = A.array(histograms).flatten()
                # this is L2 normalization (sqrt(x^2+y^2+...))
                f_norm = A.linalg.norm(f, 2)
                # cap at 0.2*norm
                for l in range(0,f.shape[0]):
                    f[l] = min(f[l], 0.2*f_norm)
                # recalcualte norm
                f_norm = A.linalg.norm(f, 2)
                # quantize to 0-255
                for l in range(0, f.shape[0]):
                    f[l] = min(A.floor(512*f[l]/f_norm), 255)
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
                distance = A.linalg.norm(keypoint_a.descriptor-keypoint_b.descriptor, 2)
                # if distance is smaller than threshold
                if(distance < sift_params.C_match_absolute):
                    distances.append(distance)
                    key_points.append(keypoint_b)
            # if only one or no keypoint is close enough
            # discard the keypoint
            if(len(distances) < 2):
                continue
            # convert to numpy array
            distances = A.array(distances)
            # find minimum distance index
            min_f_b_index = A.argmin(distances)
            # and extract distance and keypoint
            min_f_b = distances[min_f_b_index]
            keypoint_b = key_points[min_f_b_index]
            # remove from distances
            distances = A.delete(distances, min_f_b_index)
            # and find second minimum distance
            min_2_f_b = A.min(distances)
            # if the ratio of the two distances is smaller than threshold
            if(min_f_b < sift_params.C_match_relative*min_2_f_b):
                # add to matches
                if(keypoint_a in matches):
                    matches[keypoint_a].append(keypoint_b)
                else:
                    matches[keypoint_a] = [keypoint_b]
        return matches
