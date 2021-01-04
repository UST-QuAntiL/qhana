from abc import *
from math import *
from kmeansClusteringAlgorithm import KMeansClusteringAlgorithm
import numpy as np


class RotationalKMeansClusteringAlgorithm(ABC, KMeansClusteringAlgorithm):
    """
    A base class for rotational KMeans clustering algorithms.
    All angles will be calculated relatively to the given base vector.
    """

    @abstractmethod
    async def perform_clustering(self):
        pass

    def __init__(self, k, max_runs, eps, base_vector=np.array([1, 0])):
        super().__init__(k, max_runs, eps)
        self.base_vector = base_vector

    @property
    def base_vector(self):
        return self.base_vector

    @base_vector.setter
    def base_vector(self, value):
        self._base_vector = value

    def _calculate_angles(self, cartesian_points):
        """
        Calculates the angles between the 2D vectors and the base vector.
        """

        angles = np.zeros(cartesian_points.shape[0])
        for i in range(0, len(angles)):
            # formula: alpha = acos( 1/(|a||b|) * a • b )
            # here: |a| = |b| = 1
            angles[i] = acos(
                self.base_vector[0] * cartesian_points[i][0] + self.base_vector[1] * cartesian_points[i][1])

        return angles
