from numpy.typing import NDArray
import numpy as np
from SIFT_Params import SIFT_Params
from typing import Tuple
class KeyPoint:
    def __init__(self, 
                 o: int,
                 s: int,
                 m: int,
                 n: int,
                 sigma: float = 0.0,
                 x: int = 0,
                 y: int = 0,
                 omega: float = 0.0,
                 theta: float = 0.0,
                 magnitude: float = 0.0,
                 descriptor: NDArray[np.float32] = np.array([])):
        """Represents Extrema / Keypoint

        Args:
            o (int): The octave - 1-based
            s (int): The scale level from Dog image - 0-based, but no transformation to 1-based needed.
            m (int): The x - coordinate in Dog image - 0-based
            n (int): The y - coordinate in Dog image - 0-based
            sigma (float, optional): The scale level in scale space - 0-based, but no transformation to 1-based needed. Defaults to 0.0.
            x (int, optional): The x - coordinate in scale space - 0-based. Defaults to 0.
            y (int, optional): the y - coordinate in scale space - 0-based. Defaults to 0.
            omega (float, optional): The scale value in scale space. Defaults to 0.0.
            theta (float, optional): The Orientation angle in radians. Defaults to 0.0.
            magnitude (float, optional): The magnitude of the orientation. Defaults to 0.0.
            descriptor (np.ndarray, optional): The descriptor feature vector. Defaults to np.array([]).
        """
        self.o = o
        self.s = s
        self.m = m
        self.n = n
        self.sigma = sigma
        self.x = x
        self.y = y
        self.omega: float = omega
        self.theta  = theta
        self.magnitude = magnitude
        self.descriptor = descriptor
    def __str__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, omega: {self.omega}, ori: {self.theta}, mag: {self.magnitude}]"

    def __repr__(self):
        return f"[m: {self.m}, n: {self.n}, o: {self.o}, s: {self.s}, x: {self.x}, y: {self.y}, sigma: {self.sigma}, omega: {self.omega}, ori: {self.theta}, mag: {self.magnitude}]"
    def find_nearest_scale(self,
                       sigmas: list[list[float]],
                       sift_params: SIFT_Params) -> Tuple[int, int]:
        min_scale = float("inf")
        min_o = -1
        min_s = -1
        for o in range(1, sift_params.n_oct+1):
            for s in range(1, sift_params.n_spo+1):
                if(abs(sigmas[o-1][s] - self.sigma) < min_scale):
                    min_scale = sigmas[o-1][s]
                    min_o = o
                    min_s = s
        return min_o, min_s