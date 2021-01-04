from abc import *


class ClusteringAlgorithm(ABC):
    """
    A base class for general clustering algorithms.
    """

    @abstractmethod
    async def perform_clustering(self):
        pass
