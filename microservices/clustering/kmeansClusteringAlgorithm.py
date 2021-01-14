"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import *
from math import *
from clusteringAlgorithm import ClusteringAlgorithm
import numpy as np
import random


def generate_random_data(amount):
    """
    Generate amount many random 2D data points and store
    it as np.array with shape = (amount, 2).
    """

    data = np.zeros((amount, 2))

    # create random float numbers per coordinate
    for i in range(0, amount):
        data[i][0] = random.uniform(-1.0, 1.0)
        data[i][1] = random.uniform(-1.0, 1.0)

    return data


def standardize(data):
    """
    Standardize all the points given in the data np.array,
    i.e. all the points will have zero mean and unit variance.
    We expect the np.array to represent a matrix with the
    coordinates of point i being data[i] = [x, y].
    Note that a copy of the data points will be created.
    """

    # create empty arrays
    data_x = np.zeros(data.shape[0])
    data_y = np.zeros(data.shape[0])
    preprocessed_data = np.zeros_like(data)

    # create x and y coordinate arrays
    for i in range(0, len(data)):
        data_x[i] = data[i][0]
        data_y[i] = data[i][1]

    # make zero mean and unit variance, i.e. standardize
    temp_data_x = (data_x - np.mean(data_x)) / np.std(data_x)
    temp_data_y = (data_y - np.mean(data_y)) / np.std(data_y)

    # create tuples to return
    for i in range(0, data.shape[0]):
        preprocessed_data[i][0] = temp_data_x[i]
        preprocessed_data[i][1] = temp_data_y[i]

    return preprocessed_data


def normalize(data):
    """
    Normalize the data, i.e. every entry of data has length 1.
    We expect the np.array to represent a matrix with the
    coordinates of point i being data[i] = [x, y].
    Note that a copy of the data points will be created.
    """

    preprocessed_data = np.zeros_like(data)

    # create tuples and normalize
    for i in range(0, data.shape[0]):
        norm = sqrt(pow(data[i][0], 2) + pow(data[i][1], 2))
        preprocessed_data[i][0] = data[i][0] / norm
        preprocessed_data[i][1] = data[i][1] / norm

    return preprocessed_data


class KMeansClusteringAlgorithm(ClusteringAlgorithm):
    """
    A base class for KMeans clustering algorithms.
    """

    @abstractmethod
    def perform_clustering(self, data):
        pass

    def __init__(self, k, max_runs, eps):
        ClusteringAlgorithm.__init__(self)
        self.k = k
        self.max_runs = max_runs
        self.eps = eps

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, value):
        self.__k = value

    @property
    def max_runs(self):
        return self.__max_runs

    @max_runs.setter
    def max_runs(self, value):
        self.__max_runs = value

    @property
    def eps(self):
        return self.__eps

    @eps.setter
    def eps(self, value):
        self.__eps = value

    @classmethod
    def _calculate_centroids(cls, centroid_mapping, old_centroids, data):
        """
        Calculates the new cartesian positions of the
        given centroids in the centroid mapping.
        Note that a copy of the data points will be created.
        """

        # create empty arrays
        centroids = np.zeros_like(old_centroids)
        cluster_k = centroids.shape[0]

        for i in range(0, cluster_k):
            sum_x = 0
            sum_y = 0
            amount = 0
            for j in range(0, centroid_mapping.shape[0]):
                if centroid_mapping[j] == i:
                    sum_x += data[j][0]
                    sum_y += data[j][1]
                    amount += 1

            # if no points assigned to centroid, take old coordinates
            if amount == 0:
                averaged_x = old_centroids[i][0]
                averaged_y = old_centroids[i][1]
            else:
                averaged_x = sum_x / amount
                averaged_y = sum_y / amount

            norm = sqrt(pow(averaged_x, 2) + pow(averaged_y, 2))
            centroids[i][0] = averaged_x / norm
            centroids[i][1] = averaged_y / norm

        return centroids

    @classmethod
    def _calculate_relative_residual(cls, old_centroid_mapping, new_centroid_mapping):
        """
        Check whether two centroid mappings are different and how different they are
        i.e. this is the convergence condition. The relative residual is the
        percentage (0-100) of how many points have a different label in the new iteration.

        E.g. for using this function: given eps = 0.05, 100 data points in total
        =>  if from one iteration to the other less than 100 * 0.05 = 5 points change their
            label, we still accept it as converged.
        """

        count_of_different_labels = 0
        amount_of_data_points = new_centroid_mapping.shape[0]

        for i in range(0, old_centroid_mapping.shape[0]):
            if old_centroid_mapping[i] != new_centroid_mapping[i]:
                count_of_different_labels += 1

        relative_residual = (count_of_different_labels / amount_of_data_points) * 100

        print('Relative Residual: ' + str(relative_residual))

        return relative_residual

    @classmethod
    def _calculate_centroid_mapping(cls, amount_of_data, k, distances):
        """
        Calculates the centroid mapping given the distances.
        I.e. we take the amount of data, k and the distances and create the
        mapping from data point to centroid, i.e. per data point we associate
        the centroid with the shortest distance. We suppose the distances
        to be in the format distance_i = [dist i to c1, dist i to c2, ...]

        We return a list with a mapping from data point indices to centroid indices,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        ...
        """

        centroid_mapping = np.zeros(amount_of_data)
        for i in range(0, amount_of_data):
            lowest_distance = distances[i * k + 0]
            lowest_distance_centroid_index = 0
            for j in range(1, k):
                if distances[i * k + j] < lowest_distance:
                    lowest_distance_centroid_index = j
            centroid_mapping[i] = lowest_distance_centroid_index

        return centroid_mapping
