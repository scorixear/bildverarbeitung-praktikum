import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv

def main():
    img = plt.imread("Woche_12/test.jpg")
    points: list[np.ndarray] = []
    # get random points of the image
    border_size = 100
    for _ in range(4):
        x = random.randint(border_size, img.shape[1]-border_size)
        y = random.randint(border_size, img.shape[0]-border_size)
        print(f"Point: {x}, {y}, Shape: {img.shape}, Border: {border_size}")
        points.append(np.array([x, y, 1]))
    # calculate true transformation
    # by rotation of 45°
    rotation_degree = np.pi/4
    rotation_matrix = np.matrix([
        [np.cos(rotation_degree), -np.sin(rotation_degree), 0],
        [np.sin(rotation_degree), np.cos(rotation_degree), 0],
        [0, 0, 1]
    ])
    # scaling by 0.5 in the x direction
    scale_factor = 0.5
    scaling_matrix = np.matrix([
        [scale_factor, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    # and translating the image to the center
    translation_matrix = np.matrix([
        [1, 0, 500],
        [0, 1, 0],
        [0, 0, 1]])
    
    true_transformation = translation_matrix @ rotation_matrix @ scaling_matrix
    
    # transform all points
    transformed_points: list[np.ndarray] = []
    for point in points:
        # this is basically cv.warpPerspective but only on a single point
        px  = (true_transformation[0,0]*point[0] + true_transformation[0,1]*point[1] + true_transformation[0,2]*point[2]) / (true_transformation[2,0]*point[0] + true_transformation[2,1]*point[1] + true_transformation[2,2]*point[2])
        py  = (true_transformation[1,0]*point[0] + true_transformation[1,1]*point[1] + true_transformation[1,2]*point[2]) / (true_transformation[2,0]*point[0] + true_transformation[2,1]*point[1] + true_transformation[2,2]*point[2])
        transformed_points.append(np.array([px, py, 1]))
    
    # convert points to numpy arrays
    np_points = np.array(points)
    np_transformed_points = np.array(transformed_points)
    
    # estimate transformation
    transformation = estimate_transformation(np_points, np_transformed_points)

    # warp image with true and estimated
    true_transformed_img = cv.warpPerspective(img, true_transformation, (img.shape[0], img.shape[1]))
    transformed_img = cv.warpPerspective(img, transformation, (img.shape[0], img.shape[1]))
    
    # print results
    print(f"True transformation:\n{true_transformation}")
    print(f"Estimated transformation:\n{transformation}")

    # and show images
    show_imgs(img, true_transformed_img, transformed_img, np_points, np_transformed_points)
    
def estimate_transformation(points: np.ndarray, transformed_points: np.ndarray) -> np.ndarray:
    # construct affine matrix and calculate SVD
    affine = construct_affine(points, transformed_points)
    _, _, vh  = np.linalg.svd(affine, full_matrices=True)
    
    # the affine matrix is the last row of vh
    affine: np.ndarray = vh[-1].reshape((3,3))
    return affine/affine[2,2]

def construct_affine(points: np.ndarray, transformed_points: np.ndarray) -> np.ndarray:
    # points should be of same shape
    assert points.shape == transformed_points.shape, "Shapes do not match"
    num_points = points.shape[0]
    
    matrices: list[np.ndarray] = []
    # for each point-pair
    for i in range(num_points):
        # get the partial affine matrix
        partial_affine = construct_affine_partial(points[i], transformed_points[i])
        #and concatenate it to the list
        matrices.append(partial_affine)
    # will extend the matrix vertically
    return np.concatenate(matrices, axis=0)

def construct_affine_partial(point: np.ndarray, transformed_point: np.ndarray) -> np.ndarray:
    # given a point pair, construct the partial affine matrix
    x: float = point[0]
    y: float = point[1]
    z: float = point[2]
    x_t: float = transformed_point[0]
    y_t: float = transformed_point[1]
    z_t: float = transformed_point[2]
    
    # defined as such
    affine_partial = np.array([
        [0.0, 0.0, 0.0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
        [z_t*x, z_t*y, z_t*z, 0.0, 0.0, 0.0, -x_t*x, -x_t*y, -x_t*z]
    ])
    
    return affine_partial                     


def show_imgs(img, true_transformed_img, transformed_img, points, transformed_points):
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(true_transformed_img)
    axs[1].set_title("True transformed")
    axs[2].imshow(transformed_img)
    axs[2].set_title("Transformed")
    
    for i in range(points.shape[0]):
        axs[0].scatter(points[i, 0], points[i, 1], c="r")
        axs[1].scatter(transformed_points[i, 0], transformed_points[i, 1], c="r")
        axs[2].scatter(transformed_points[i, 0], transformed_points[i, 1], c="r")
    plt.show()

if __name__ == "__main__":
    main()
