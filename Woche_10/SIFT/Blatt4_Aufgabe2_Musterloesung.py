from typing import Tuple
import cv2 as cv
import numpy as np
from numpy.typing import NDArray # if code cannot be run replace NDArray[np.float32] with np.ndarray

# custom imports
from SIFT_algo import SIFT_Algorithm # implements the SIFT algorithm
from SIFT_KeyPoint import SIFT_KeyPoint # represents a keypoint or extremum found
from SIFT_Params import SIFT_Params # represents all hyperparameters for the algorithm
from SIFT_Visualization import visualize_scale_space, visualize_keypoints # visualizes the results
    
def detect_and_compute(img: NDArray[np.float32],
                       sift_params: SIFT_Params,
                       show_plots: bool = False) -> Tuple[list[list[NDArray[np.float32]]], list[list[NDArray[np.float32]]], list[SIFT_KeyPoint]]:
    """detects and computes keypoints and descriptors for a given image

    Args:
        img (NDArray[np.float32]): the image to detect keypoints and descriptors for
        sift_params (SIFT_Params): the parameters for the SIFT algorithm
        show_plots (bool, optional): Shows Matplotlib plots for each step. Defaults to False.

    Returns:
        Tuple[list[list[NDArray[np.float32]]], list[list[NDArray[np.float32]]], list[KeyPoint]]: scale space [octave][scale], dogs [octave, scale-1], keypoints
    """
    # Image scaled down to lowest octave must have at least 12 pixel width and height
    if (img.shape[0] < 2**(sift_params.n_oct-1)*12 or img.shape[1] < 2**(sift_params.n_oct-1)*12):
        raise ValueError("Image is too small for the given number of octaves")
    
    scale_space, deltas, sigmas = SIFT_Algorithm.create_scale_space(img, sift_params)
    dogs = SIFT_Algorithm.create_dogs(scale_space, sift_params)
    # normalize dogs are just for visualization
    normalized_dogs = [[cv.normalize(d, None, 0, 1, cv.NORM_MINMAX) for d in octave] for octave in dogs] # type: ignore
    discrete_extremas = SIFT_Algorithm.find_discrete_extremas(dogs, sift_params, sigmas, deltas)
    taylor_extremas = SIFT_Algorithm.taylor_expansion(discrete_extremas,
                                       dogs,
                                       sift_params,
                                       deltas,
                                       sigmas)
    filtered_extremas = SIFT_Algorithm.filter_extremas(taylor_extremas,
                                        dogs,
                                        sift_params)
    gradients = SIFT_Algorithm.gradient_2d(scale_space, sift_params)
    key_points = SIFT_Algorithm.assign_orientations(filtered_extremas,
                                     scale_space,
                                     sift_params,
                                     gradients,
                                     deltas)
    descriptor_key_points = SIFT_Algorithm.create_descriptors(key_points,
                                               scale_space,
                                               sift_params,
                                               gradients,
                                               deltas)
    
    if(show_plots):
        visualize_scale_space(scale_space, "Scale Space")
        visualize_scale_space(dogs, "Normalized DoGs")
        visualize_keypoints(dogs, discrete_extremas, deltas, "Discrete Extremas")
        visualize_keypoints(dogs, taylor_extremas, deltas, "Taylor Extremas")
        visualize_keypoints(dogs, filtered_extremas, deltas, "Filtered Extremas")
        visualize_keypoints(scale_space, key_points, deltas, "Key Points", True, True)
        visualize_keypoints(scale_space, descriptor_key_points, deltas, "Descriptor Key Points", True, True, True)
    
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
    img = cv.imread("Woche_10/cat.jpg")
    # resize image for faster processing
    img = cv.resize(img, (128, 104))
    # convert to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = (gray/255.).astype(np.float32)
    rotated = rotate_image(gray, 45)
    sift_params = SIFT_Params()
    _, _, keypoints = detect_and_compute(gray, sift_params, show_plots=True)
    _, _, rotated_keypoints = detect_and_compute(rotated, sift_params, show_plots=True)

    # Keypoint Matching
    matched_keypoints = SIFT_Algorithm.match_keypoints(keypoints, rotated_keypoints, sift_params)
    print(matched_keypoints)
    
if __name__ == "__main__":
    main()
