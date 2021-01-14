"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import *


class ClusteringAlgorithm(ABC):
    """
    A base class for general clustering algorithms.
    """

    def __init__(self):
        pass

    @abstractmethod
    def perform_clustering(self, data):
        pass
