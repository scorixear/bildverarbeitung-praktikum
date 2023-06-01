from enum import Enum
import numpy as np

class Axis(Enum):
    """Available Axis choices

    Args:
        Enum (int): the axis
    """
    X = 0
    Y = 1
    Z = 2

class Homogene3D:
    """represents 3D coordinates in homogene coordinates
    """
    def __init__(self, vector: list[float] | None =None) -> None:
        """Initializes the Homogene3D coordinate

        Args:
            vector (list[float], optional): if given, should have 3 values. Defaults to None.
        """
        if(vector is not None):
            if(len(vector) != 3):
                raise ValueError("vector must have 3 values")
            self.coordinate: np.matrix = np.matrix([vector[0], vector[1], vector[2], 1]).transpose()
        
    @staticmethod
    def translation(translation_vector: list[float]) -> np.matrix:
        """Translates the coordinate system by the given vector

        Args:
            translation_vector (list[float]): the translation values

        Returns:
            np.matrix: the translation matrix
        """
        assert len(translation_vector) == 3
        return np.matrix([
            [1, 0, 0, translation_vector[0]],
            [0, 1, 0, translation_vector[1]],
            [0, 0, 1, translation_vector[2]],
            [0, 0, 0, 1]])
    @staticmethod
    def rotation(theta: float, axis: Axis) -> np.matrix:
        """Rotation by the given angle around the given axis

        Args:
            theta (float): the angle in radians
            axis (Axis): the axis to rotate around

        Raises:
            ValueError: if the axis is not x, y or z

        Returns:
            np.matrix: the rotation matrix
        """
        if axis == Axis.X:
            return np.matrix([
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]])
        if axis == Axis.Y:
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]])
        if axis == Axis.Z:
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
        raise ValueError("Axis must be x, y or z")
    @staticmethod
    def affine(matrix: np.matrix, vector: list[float]) -> np.matrix:
        """Affine transformation matrix

        Args:
            matrix (np.matrix): the matrix for affine transformation
            vector (list[float]): the vector for translation

        Returns:
            np.matrix: the affine transformation matrix
        """
        assert len(vector) == 3
        assert matrix.shape == (3,3)
        return np.matrix([
            [matrix[0,0], matrix[0,1], matrix[0,2], vector[0]],
            [matrix[1,0], matrix[1,1], matrix[1,2], vector[1]],
            [matrix[2,0], matrix[2,1], matrix[2,2], vector[2]],
            [0, 0, 0, 1]])
    @staticmethod
    def scale(scale_vector: list[float]) -> np.matrix:
        """Scaling matrix

        Args:
            scale_vector (list[float]): the scaling values

        Returns:
            np.matrix: the matrix for scaling
        """
        assert len(scale_vector) == 3
        return np.matrix([
            [scale_vector[0], 0, 0, 0],
            [0, scale_vector[1], 0, 0],
            [0, 0, scale_vector[2], 0],
            [0, 0, 0, 1]])
    
    def apply_transformation(self, *transformation_matricies: np.matrix) -> "Homogene3D":
        """Applies multiple transformation in the given order (0 first, n last)

        Raises:
            ValueError: if transformation matrix is not 4x4

        Returns:
            Homogene3D: the new coordinates
        """
        current = self.coordinate
        for transformation_matrix in transformation_matricies:
            if transformation_matrix.shape != (4,4):
                raise ValueError("Transformation matrix must be 4x4")
            current = np.matmul(transformation_matrix, current)
        new_coordinates = Homogene3D()
        new_coordinates.coordinate = current
        return new_coordinates
    def __str__(self) -> str:
        return str(self.coordinate)


def main():
    # create given point
    point = Homogene3D([3,7,2])
    # generate given transformation matricies
    translation = Homogene3D.translation([5,2,2])
    rotation = Homogene3D.rotation(np.pi/2, Axis.Y)
    scale = Homogene3D.scale([3,3,3])
    # apply transformation
    new_point = point.apply_transformation(translation, rotation, scale)
    # print out results
    print(point)
    print(new_point)

if __name__ == "__main__":
    main()