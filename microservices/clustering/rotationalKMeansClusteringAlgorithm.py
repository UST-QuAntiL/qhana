"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from kmeansClusteringAlgorithm import *
import numpy as np


class RotationalKMeansClusteringAlgorithm(KMeansClusteringAlgorithm):
    """
    A base class for rotational KMeans clustering algorithms.
    All angles will be calculated relatively to the given base vector.
    """

    def __init__(self, k, max_runs, eps, base_vector=np.array([1, 0])):
        KMeansClusteringAlgorithm.__init__(self, k, max_runs, eps)
        self.base_vector = base_vector

    @property
    def base_vector(self):
        return self.__base_vector

    @base_vector.setter
    def base_vector(self, value):
        self.__base_vector = value

    @abstractmethod
    def _perform_rotational_clustering(self, centroid_angles, data_angles):
        pass

    def perform_clustering(self, data):
        """
        Executes a quantum k means cluster algorithm with using the
        abstract perform_rotational_clustering method.
        The data needs to be 2D cartesian representation in
        an np.array of shape = (amount, 2).

        We return a np.array with a mapping from data indices to centroid indices,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        """

        # generate k initial random centroids
        centroids_raw = generate_random_data(self.k)

        # map to unit sphere
        centroids = normalize(standardize(centroids_raw))
        preprocessed_data = normalize(standardize(data))

        # data angles don't need to be updated in every iteration,
        # they are fixed
        data_angles = self._calculate_angles(preprocessed_data)

        # prepare initial data before entering loop
        new_centroid_mapping = np.zeros(preprocessed_data.shape[0])
        counter = 1
        converged = False

        while not converged and counter < self.max_runs:
            print("Run " + str(counter) + " from " + str(self.max_runs))
            centroid_angles = self._calculate_angles(centroids)
            old_centroid_mapping = np.copy(new_centroid_mapping)

            new_centroid_mapping = self._perform_rotational_clustering(centroid_angles, data_angles)

            relative_residual = self._calculate_relative_residual(old_centroid_mapping, new_centroid_mapping)
            converged = relative_residual < self.eps

            if not converged:
                centroids = self._calculate_centroids(new_centroid_mapping, centroids, preprocessed_data)
                centroids = normalize(centroids)
                counter += 1

        return new_centroid_mapping

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
