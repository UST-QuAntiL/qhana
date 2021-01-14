"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import numpy as np
from sklearn.cluster import KMeans
from kmeansClusteringAlgorithm import KMeansClusteringAlgorithm


class SklearnClustering(KMeansClusteringAlgorithm):

    def __init__(self, k, max_runs, eps):
        KMeansClusteringAlgorithm.__init__(self, k, max_runs, eps)

    def perform_clustering(self, data):
        """
        Performs the sklearn KMeans clustering algorithm.

        We return a np.array with a mapping from data indices to centroid indices,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1

        Note, that we use the sklearn residual condition which is
        the Frobenius norm for the calculated centroids.
        """

        # apply kmeans classically
        kmeans_algorithm = KMeans(n_clusters=self.k, random_state=0, max_iter=self.max_runs, tol=self.eps)
        centroid_mapping = kmeans_algorithm.fit(data).labels_.astype(np.int)
        return centroid_mapping
