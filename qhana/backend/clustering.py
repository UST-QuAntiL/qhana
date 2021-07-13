from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import numpy as np
from backend.logger import Logger, LogLevel
from sklearn.cluster import OPTICS, KMeans
from sklearn_extra.cluster import KMedoids
from backend.entity import Costume
from typing import List
from backend.quantumKMeans import NegativeRotationQuantumKMeans, DestructiveInterferenceQuantumKMeans, \
    StatePreparationQuantumKMeans, PositiveCorrelationQuantumKmeans

from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.optimization.applications.ising import max_cut
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua import QuantumInstance
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit import IBMQ

import networkx as nx
from backend.classicNaiveMaxCutSolver import ClassicNaiveMaxCutSolver
from backend.sdpMaxCutSolver import SdpMaxCutSolver
from backend.bmMaxCutSolver import BmMaxCutSolver
from backend.timer import Timer

"""
Enums for Clustertyps
"""
class ClusteringType(enum.Enum):
    optics = 0  # OPTICS =:Ordering Points To Identify the Clustering Structure
    vqeMaxCut = 1 # MaxCut Quantenalgorithmus based on VQE 
    classicNaiveMaxCut = 2 # Naive classical implementation
    sdpMaxCut = 3 # Semidefinite Programming implementation
    bmMaxCut = 4 # Bureir-Monteiro implementation
    negativeRotationQuantumKMeans = 5 # Negative Rotation Quantum K Means
    destructiveInterferenceQuantumKMeans = 6 # Destructive Interference Quantum K Means
    ClassicalKMeans = 7 # a classical scikit implementation of K means
    StatePreparationQuantumKMeans = 8 # an own implementation of a quantum k means
    PositiveCorrelationQuantumKMeans = 9
    ClassicalKMedoids = 10

    @staticmethod
    def get_name(clusteringTyp) -> str:
        name = ""
        if clusteringTyp == ClusteringType.optics :
            name = "optics"
        elif clusteringTyp == ClusteringType.vqeMaxCut :
            name = "qaoaMaxCut"
        elif clusteringTyp == ClusteringType.classicNaiveMaxCut:
            name = "classicNaiveMaxCut"
        elif clusteringTyp == ClusteringType.sdpMaxCut:
            name = "sdpMaxCut"
        elif clusteringTyp == ClusteringType.bmMaxCut:
            name = "bmMaxCut"
        elif clusteringTyp == ClusteringType.negativeRotationQuantumKMeans:
            name = "negativeRotationQuantumKMeans"
        elif clusteringTyp == ClusteringType.destructiveInterferenceQuantumKMeans:
            name = "destructiveInterferenceQuantumKMeans"
        elif clusteringTyp == ClusteringType.ClassicalKMeans:
            name = "classicalKMeans"
        elif clusteringTyp == ClusteringType.StatePreparationQuantumKMeans:
            name = "statePreparationQuantumKMeans"
        elif clusteringTyp == ClusteringType.PositiveCorrelationQuantumKMeans:
            name = "positiveCorrelationQuantumKMeans"
        elif clusteringTyp == ClusteringType.ClassicalKMedoids:
            name = "classicalKMedoids"
        else:
            Logger.error("No name for clustering \"" + str(clusteringTyp) + "\" specified")
            raise ValueError("No name for clustering \"" + str(clusteringTyp) + "\" specified")
        return name

    """
    Returns the description of the given ScalingType.
    """
    @staticmethod
    def get_description(clusteringTyp) -> str:
        description = ""
        if clusteringTyp == ClusteringType.optics:
            description = ("Ordering points to identify the clustering structure (OPTICS)" 
                         + " is an algorithm for finding density-based clusters"
                         + " in spatial data")
        elif clusteringTyp == ClusteringType.vqeMaxCut:
            description = ("MaxCut Quantum Algorithm based on QAOA")
        elif clusteringTyp == ClusteringType.classicNaiveMaxCut:
            description = ("Classical naive implemented MaxCut algorithm")
        elif clusteringTyp == ClusteringType.sdpMaxCut:
            description = ("Semidefinite Programming solver for MaxCut")
        elif clusteringTyp == ClusteringType.bmMaxCut:
            description = ("Bureir-Monteiro solver for MaxCut")
        elif clusteringTyp == ClusteringType.negativeRotationQuantumKMeans:
            description = ("Negative Rotation Quantum K Means")
        elif clusteringTyp == ClusteringType.destructiveInterferenceQuantumKMeans:
            description = ("Destructive Interference Quantum K Means")
        elif clusteringTyp == ClusteringType.ClassicalKMeans:
            description = ("Classical K Means")
        elif clusteringTyp == ClusteringType.StatePreparationQuantumKMeans:
            description = ("State Preparation Quantum K Means")
        elif clusteringTyp == ClusteringType.PositiveCorrelationQuantumKMeans:
            description = ("Positive Correlation Quantum K Means")
        elif clusteringTyp == ClusteringType.ClassicalKMedoids:
            description = ("Classical K Medoids")
        else:
            Logger.error("No description for clustering \"" + str(clusteringTyp) + "\" specified")
            raise ValueError("No description for clustering \"" + str(clusteringTyp) + "\" specified")
        return description


class Clustering(metaclass=ABCMeta):

    def __init__(self):
        self.keep_cluster_mapping = False

    """
    Interface for Clustering Object
    """
    @abstractmethod
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        pass
    
    @abstractmethod
    def get_param_list(self) -> list:
        pass

    @abstractmethod
    def set_param_list(self, params: list = []) -> np.matrix:
        pass

    @abstractmethod
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass

    def get_keep_cluster_mapping(self):
        return self.keep_cluster_mapping

    def set_keep_cluster_mapping(self, keep_cluster_mapping):
        self.keep_cluster_mapping = keep_cluster_mapping
        return

""" 
Represents the factory to create an scaling object
"""
class ClusteringFactory:
    
    @staticmethod
    def create(type: ClusteringType) -> Clustering:
        if type == ClusteringType.optics:
            return Optics()
        elif type == ClusteringType.vqeMaxCut:
            return VQEMaxCut()
        elif type == ClusteringType.classicNaiveMaxCut:
            return ClassicNaiveMaxCut()
        elif type == ClusteringType.sdpMaxCut:
            return SdpMaxCut()
        elif type == ClusteringType.bmMaxCut:
            return BmMaxCut()
        elif type == ClusteringType.negativeRotationQuantumKMeans:
            return NegativeRotationQuantumKMeansClustering()
        elif type == ClusteringType.destructiveInterferenceQuantumKMeans:
            return DestructiveInterferenceQuantumKMeansClustering()
        elif type == ClusteringType.ClassicalKMeans:
            return ClassicalKMeans()
        elif type == ClusteringType.StatePreparationQuantumKMeans:
            return StatePreparationQuantumKMeansClustering()
        elif type == ClusteringType.PositiveCorrelationQuantumKMeans:
            return PositiveCorrelationQuantumKMeansClustering()
        elif type == ClusteringType.ClassicalKMedoids:
            return ClassicalKMedoids()
        else:
            Logger.error("Unknown type of clustering. The application will quit know.")
            raise Exception("Unknown type of clustering.")


"""
optics
"""
class Optics(Clustering):
    """
    OPTICS Referenz : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

        OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, finds
        core sample of high density and expands clusters from them [R2c55e37003fe-1]. Unlike DBSCAN, k-
        eeps cluster hierarchy for a variable neighborhood radius. Better suited for usage on large da-
        tasets than the current sklearn implementation of DBSCAN.
        Clusters are then extracted using a DBSCAN-like method (cluster_method = ‘dbscan’) or an automa-
        tic technique proposed in [R2c55e37003fe-1] (cluster_method = ‘xi’).
        This implementation deviates from the original OPTICS by first performing k-nearest-neighborhood
        searches on all points to identify core sizes, then computing only the distances to unprocessed 
        points when constructing the cluster order. Note that we do not employ a heap to manage the exp-
        ansion candidates, so the time complexity will be O(n^2). Read more in the User Guide.

     Parameters:
        min_samples:        int > 1 or float between 0 and 1 (default=5)
                            The number of samples in a neighborhood for a point to be considered as a
                            core point. Also, up and down steep regions can’t have more then min_sam-
                            ples consecutive non-steep points. Expressed as an absolute number or a 
                            fraction of the number of samples (rounded to be at least 2).

        max_eps:            float, optional (default=np.inf)
                            The maximum distance between two samples for one to be considered as in t-
                            he neighborhood of the other. Default value of np.inf will identify clust-
                            ers across all scales; reducing max_eps will result in shorter run times.

        metric:             str or callable, optional (default=’minkowski’)
                            Metric to use for distance computation. Any metric from scikit-learn or 
                            scipy.spatial.distance can be used.

                            If metric is a callable function, it is called on each pair of instances 
                            (rows) and the resulting value recorded. The callable should take two arr-
                            ays as input and return one value indicating the distance between them. 
                            This works for Scipy’s metrics, but is less efficient than passing the me-
                            tric name as a string. If metric is “precomputed”, X is assumed to be a 
                            distance matrix and must be square.

                            Valid values for metric are:

                            from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, 
                            ‘manhattan’]

                            from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, 
                            ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, 
                            ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
                            ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

                            See the documentation for scipy.spatial.distance for details on these metrics.

        p:                  int, optional (default=2)
                            Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. 
                            When p = 1, this is equivalent to using manhattan_distance (l1), and euclidea-
                            n_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric_params:      dict, optional (default=None)
                            Additional keyword arguments for the metric function.

        cluster_method:     str, optional (default=’xi’)
                            The extraction method used to extract clusters using the calculated reachability
                            and ordering. Possible values are “xi” and “dbscan”.

        eps:                float, optional (default=None)
                            The maximum distance between two samples for one to be considered as in the neig-
                            hborhood of the other. By default it assumes the same value as max_eps. Used only 
                            when cluster_method='dbscan'.

        xi:                 float, between 0 and 1, optional (default=0.05)
                            Determines the minimum steepness on the reachability plot that constitutes a clu-
                            ster boundary. For example, an upwards point in the reachability plot is defined 
                            by the ratio from one point to its successor being at most 1-xi. Used only when 
                            cluster_method='xi'.

     predecessor_correction:bool, optional (default=True)
                            Correct clusters according to the predecessors calculated by OPTICS [R2c55e37003fe-2].
                            This parameter has minimal effect on most datasets. Used only when cluster_method='xi'.

        min_cluster_size:   int > 1 or float between 0 and 1 (default=None)
                            Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a 
                            fraction of the number of samples (rounded to be at least 2). If None, the value of 
                            min_samples is used instead. Used only when cluster_method='xi'.

        algorithm:          {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
                            Algorithm used to compute the nearest neighbors:

                            ‘ball_tree’ will use BallTree
                            ‘kd_tree’ will use KDTree
                            ‘brute’ will use a brute-force search.
                            ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed
                             to fit method. (default)

                            Note: fitting on sparse input will override the setting of this parameter, using brute 
                            force.

        leaf_size:          int, optional (default=30)
                            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction 
                            and query, as well as the memory required to store the tree. The optimal value depends on
                            the nature of the problem.

        n_jobs:             int or None, optional (default=None)
                            The number of parallel jobs to run for neighbors search. None means 1 unless in a 
                            joblib.parallel_backend context. -1 means using all processors. See Glossary for more det-
                            ails.
    """
    def __init__(
        self,
        min_samples: float = 5, # float between 0 and 1 else int
        max_eps: float = np.inf,
        metric: str = 'minkowski',
        p: int = 2,   #only when the minkowski metric is choosen
        metric_params: dict = None, # additional keywords for the metric function
        cluster_method: str = 'xi',
        eps: float = None,  # by default it assumes the same value as max_eps (Only used when cluster_method='dbscan')
        xi: float = 0.05, # only between 0 and 1 (Only used when cluster_method='xi')
        predecessor_correction: bool = True, # only used when cluster_method='xi'
        min_cluster_size: float = None, # float between 0 and 1 else int
        algorithm: str = 'auto',
        leaf_size: int = 30, # only for BallTree or KDTree
        n_jobs: int = None # -1 mean using all processors
    ):
        super().__init__()
        if min_samples <= 1 and min_samples >= 0:
            self.__min_samples: float = min_samples
        elif min_samples > 1:
            self.__min_samples: int = round(min_samples)
        else:
            Logger.error("min_samples is smaller than 0.")
            raise Exception("min_samples is smaller than 0")
        self.__max_eps: float = max_eps
        self.__metric: str = metric
        self.__p: int = p
        self.__metric_params: dict = None
        self.__cluster_method: str = cluster_method
        self.__eps: float = eps
        if xi >= 0 and xi <= 1:
            self.__xi : float = xi 
        else:
            Logger.warning("xi is not between 0 and 1. Default Value was set! xi = 0.05")
            self.__xi : float = 0.05
        self.__predecessor_correction: bool = predecessor_correction
        self.__algorithm: str = algorithm
        self.__leaf_size: int = leaf_size
        self.__n_jobs: int = n_jobs
        if  min_cluster_size == None or (min_cluster_size >= 0 and min_cluster_size <= 1):
            self.__min_cluster_size : float = min_cluster_size 
        else:
            Logger.warning("min is not between 0 and 1 or None. Default Value was set! min_cluster_size = None")
            self.__min_cluster_size : float = None
        
        try:
            self.__cluster_instance: OPTICS = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

        # sklearn.cluster._optics.OPTICS

    def __create_optics_cluster_instance(self) -> OPTICS :
        if self.__min_samples < 0:
            Logger.error("min_samples is smaller than 0.")
            raise Exception("min_samples is smaller than 0")
        elif self.__min_samples > 1:
            self.__min_samples = round(self.__min_samples)
                
        if self.__cluster_method != "xi" and self.__cluster_method != "dbscan":
            Logger.error("Not valid cluster_method.")
            raise Exception("Not valid cluster_method.")
        
        if  self.__min_cluster_size  != None and self.__min_cluster_size  < 0 and self.__min_cluster_size > 1 :
            Logger.warning("min is not between 0 and 1 or None. Default Value was set! min_cluster_size = None")
            self.__min_cluster_size : float = None
        
        if self.__algorithm != "auto" and self.__algorithm != "ball_tree" and self.__algorithm != "kd_tree" and self.__algorithm != "brute":
            Logger.error("Not valid algorithm method.")
            raise Exception("Not valid algorithm method.")

        if self.__cluster_method == "xi":
            if self.__xi > 1 and self.__xi < 0:
                Logger.warning("xi is not between 0 and 1. Default Value was set! xi = 0.05")
                self.__xi : float = 0.05

            if self.__algorithm == "ball_tree" or self.__algorithm == "kd_tree":

                if self.__metric == "minkowski":
                    # xi, ball algorithm , minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    p                       = self.__p,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    xi                      = self.__xi,
                                    predecessor_correction  = self.__predecessor_correction,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    leaf_size               = self.__leaf_size,
                                    n_jobs                  = self.__n_jobs
                                )
                else:
                    # xi, ball algorithm , not minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    xi                      = self.__xi,
                                    predecessor_correction  = self.__predecessor_correction,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    leaf_size               = self.__leaf_size,
                                    n_jobs                  = self.__n_jobs
                                )
            else:
                if self.__metric == "minkowski":
                    # xi, not ball algorithm, minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    p                       = self.__p,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    xi                      = self.__xi,
                                    predecessor_correction  = self.__predecessor_correction,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    n_jobs                  = self.__n_jobs
                                )
                else:
                    # xi, not ball algorithm , not minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    xi                      = self.__xi,
                                    predecessor_correction  = self.__predecessor_correction,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    n_jobs                  = self.__n_jobs
                                )


        elif self.__cluster_method == "dbscan":
            if self.__algorithm == "ball_tree" or self.__algorithm == "ball_tree":

                if self.__metric == "minkowski":
                    # dbscan, ball algorithm , minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    p                       = self.__p,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    eps                     = self.__eps,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    leaf_size               = self.__leaf_size,
                                    n_jobs                  = self.__n_jobs
                                )
                else:
                    # dbscan, ball algorithm , not minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    eps                     = self.__eps,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    leaf_size               = self.__leaf_size,
                                    n_jobs                  = self.__n_jobs
                                )

            else:
                if self.__metric == "minkowski":
                    # dbscan, not ball algorithm, minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    p                       = self.__p,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    eps                     = self.__eps,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    n_jobs                  = self.__n_jobs
                                )
                else:
                    # dbscan, not ball algorithm , not minkowski
                    return OPTICS(  min_samples             = self.__min_samples,
                                    max_eps                 = self.__max_eps,
                                    metric                  = self.__metric,
                                    metric_params           = self.__metric_params,
                                    cluster_method          = self.__cluster_method,
                                    eps                     = self.__eps,
                                    min_cluster_size        = self.__min_cluster_size,
                                    algorithm               = self.__algorithm,
                                    n_jobs                  = self.__n_jobs
                                )

    def create_cluster(self, position_matrix : np.matrix, similarity_matrix : np.matrix ) -> np.matrix:
        try:
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

        try: 
            self.__cluster_instance.fit(position_matrix)
            return self.__cluster_instance.labels_  
        except Exception as error:
            Logger.error("An Exception occurs by clustering the postion_matrix: " + str(error))
            raise Exception("Exception occurs in Method create_cluster by clustering the positon_matrix.")        
  
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass

    # getter methodes
    def get_min_samples(self) -> float:
        return self.__min_samples
    
    def get_max_eps(self) -> float:
        return self.__max_eps
    
    def get_metric(self) -> str:
        return self.__metric    
    
    def get_p(self) -> int:
        return self.__p
    
    def get_metric_params(self) -> dict:
        return self.__metric_params
    
    def get_cluster_method(self) -> str:
        return self.__cluster_method
    
    def get_eps(self) -> float:
        return self.__eps
    
    def get_xi(self) -> float:
        return self.__xi
    
    def get_predecessor_correction(self) -> bool:
        return self.__predecessor_correction    
    
    def get_min_cluster_size(self) -> float:
        return self.__min_cluster_size
    
    def get_algorithm(self) -> str:
        return self.__algorithm
    
    def get_leaf_size(self) -> int:
        return self.__leaf_size
    
    def get_n_jobs(self) -> int:
        return self.__n_jobs
    
    def get_cluster_instance(self) -> OPTICS:
        return self.__cluster_instance

    # setter methodes
    def set_min_samples(self, min_samples: float = 5) -> None:
        try:
            self.__min_samples = min_samples
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

    def set_max_eps(self, max_eps: float = np.inf) -> None:
        try:
            self.__max_eps = max_eps
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_metric(self, metric: str = 'minkowski') -> None:
        try:
            self.__metric = metric
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_p(self, p: int = 2) -> None:
        try:
            self.__p = p
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_metric_params(self, metric_params: dict = None) -> None:
        try:
            self.__metric_params = metric_params
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
        
    def set_cluster_method(self, cluster_method: str = 'xi') -> None:
        try:
            self.__cluster_method = cluster_method
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_eps(self, eps: float = None) -> None:
        try:
            self.__eps = eps
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_xi(self, xi: float = 0.05) -> None:
        try:
            self.__xi = xi
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
        
    def set_predecessor_correction(self, predecessor_correction: bool = True) -> None:
        try:
            self.__predecessor_correction = predecessor_correction
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_min_cluster_size(self, min_cluster_size: float = None) -> None:
        try:
            self.__min_cluster_size = min_cluster_size
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_algorithm(self, algorithm: str = 'auto') -> None:
        try:
            self.__algorithm = algorithm
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_leaf_size(self, leaf_size: int = 30) -> None:
        try:
            self.__leaf_size = leaf_size
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")
    
    def set_n_jobs(self, n_jobs: int = None) -> None:
        try:
            self.__n_jobs = n_jobs
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            Logger.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

    # setter and getter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "OPTICS"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_minSamples = self.get_min_samples()
        description_minSamples = "int > 1 or float between 0 and 1 (default=5)"\
                            +"The number of samples in a neighborhood for a point to be considered as a "\
                            +"core point. Also, up and down steep regions can’t have more then min_samples "\
                            +"consecutive non-steep points. Expressed as an absolute number or a " \
                            +"fraction of the number of samples (rounded to be at least 2)."
        params.append(("minSamples", "min Samples" ,description_minSamples, parameter_minSamples, "number", 1 , 1 ))
        
        parameter_maxEps = self.get_max_eps()
        description_maxEps = "float, optional (default=np.inf) "\
                            +"The maximum distance between two samples for one to be considered as in "\
                            +"the neighborhood of the other. Default value of np.inf will identify clusters "\
                            +"across all scales; reducing max_eps will result in shorter run times."
        params.append(("maxEps", "max Epsilon" ,description_maxEps, parameter_maxEps, "text" ))
        
        parameter_metric = self.get_metric()
        description_metric = "str or callable, optional (default=’minkowski’) "\
                            +"Metric to use for distance computation. Any metric from scikit-learn or "\
                            +"scipy.spatial.distance can be used."
        params.append(("metric", " Metric" ,description_metric, parameter_metric , "select" , ('precomputed','minkowski','cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan')))
        
        parameter_p = self.get_p()
        description_p = "int, optional (default=2) "\
                            +"Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. "\
                            +"When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean"\
                            +"_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used."
        params.append(("p" , "Parameter p for minkowski" ,description_p, parameter_p, "number" , 1,1 ))
        
        parameter_cluster_method = self.get_cluster_method()
        description_cluster_method = "str, optional (default=’xi’) "\
                                    +"The extraction method used to extract clusters using the calculated reachability "\
                                    +"and ordering. Possible values are “xi” and “dbscan”."
        params.append(("cluster_method","Cluster Method",description_cluster_method, parameter_cluster_method, "select" , ("xi" , "dbscan")))
        
        parameter_eps = self.get_eps()
        description_eps =    "float, optional (default=None) "\
                            +"The maximum distance between two samples for one to be considered as in the "\
                            +"neighborhood of the other. By default it assumes the same value as max_eps. Used only "\
                            +"when cluster_method='dbscan'."
        params.append(("eps", "Epsilon",description_eps, parameter_eps , "text"))
        
        parameter_xi = self.get_xi()
        description_xi = "float, between 0 and 1, optional (default=0.05) "\
                            +"Determines the minimum steepness on the reachability plot that constitutes a cluster "\
                            +"boundary. For example, an upwards point in the reachability plot is defined " \
                            +"by the ratio from one point to its successor being at most 1-xi. Used only when "\
                            +"cluster_method='xi'."
        params.append(("xi","Xi" ,description_xi, parameter_xi, "number" , 0, 0.001))
        
        parameter_predecessor_correction = self.get_predecessor_correction()
        description_predecessor_correction = "bool, optional (default=True) "\
                            +"Correct clusters according to the predecessors calculated by OPTICS [R2c55e37003fe-2]. "\
                            +"This parameter has minimal effect on most datasets. Used only when cluster_method='xi'."
        params.append(("predecessor_correction", "Predecessor Correction" ,description_predecessor_correction, parameter_predecessor_correction, "checkbox"))
        
        parameter_min_cluster_size = self.get_min_cluster_size()
        description_min_cluster_size = "int > 1 or float between 0 and 1 (default=None) "\
                            +"Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a "\
                            +"fraction of the number of samples (rounded to be at least 2). If None, the value of "\
                            +"min_samples is used instead. Used only when cluster_method='xi'."
        params.append(("min_cluster_size","Min Cluster Size",description_min_cluster_size, parameter_min_cluster_size, "text"))
        
        parameter_algorithm = self.get_algorithm()
        description_algorithm = "{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional "\
                            +"Algorithm used to compute the nearest neighbors: "\
                            +"‘ball_tree’ will use BallTree "\
                            +"‘kd_tree’ will use KDTree "\
                            +"‘brute’ will use a brute-force search. "\
                            +"‘auto’ will attempt to decide the most appropriate algorithm based on the values passed "\
                            +"to fit method. (default)"
        params.append(("algorithm","Algorithm" ,description_algorithm, parameter_algorithm , "select" , ('auto', 'ball_tree', 'kd_tree', 'brute')))
        
        parameter_leaf_size = self.get_leaf_size()
        description_leaf_size = "int, optional (default=30) "\
                            +"Leaf size passed to BallTree or KDTree. This can affect the speed of the construction "\
                            +"and query, as well as the memory required to store the tree. The optimal value depends on "\
                            +"the nature of the problem."
        params.append(("leaf_size","Leaf Size" ,description_leaf_size, parameter_leaf_size, "number", 0 , 1))
        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()

        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params

    def set_param_list(self, params: list = []) -> np.matrix:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        for param in params:
            if param[0] == "minSamples":
                self.set_min_samples(param[3])
            elif param[0] == "maxEps":
                self.set_max_eps(param[3])
            elif param[0] == "metric":
                self.set_metric(param[3])
            elif param[0] == "p":
                self.set_p(param[3])
            elif param[0] == "cluster_method":
                self.set_cluster_method(param[3])
            elif param[0] == "eps":
                self.set_eps(param[3])
            elif param[0] == "xi":
                self.set_xi(param[3])
            elif param[0] == "predecessor_correction":
                self.set_predecessor_correction(param[3])
            elif param[0] == "min_cluster_size":
                self.set_min_cluster_size(param[3])
            elif param[0] == "algorithm":
                self.set_algorithm(param[3])
            elif param[0] == "leaf_size":
                self.set_leaf_size(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])


class QuantumBackends(enum.Enum):
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_16_melbourne = "ibmq_16_melbourne"
    ibmq_armonk = "ibmq_armonk"
    ibmq_5_yorktown = "ibmq_5_yorktown"
    ibmq_ourense = "ibmq_ourense"
    ibmq_vigo = "ibmq_vigo"
    ibmq_valencia = "ibmq_valencia"
    ibmq_athens = "ibmq_athens"
    ibmq_santiago = "ibmq_santiago"

    @staticmethod
    def get_quantum_backend(backendEnum, ibmqToken = None, customBackendName = None):
        backend = None
        if backendEnum.name.startswith("aer"):
            # Use local AER backend
            aerBackendName = backendEnum.name[4:]
            backend = Aer.get_backend(aerBackendName)
        elif backendEnum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmqToken)
            backend = provider.get_backend(backendEnum.name)
        elif backendEnum.name.startswith("custom_ibmq"):
            provider = IBMQ.enable_account(ibmqToken)
            backend = provider.get_backend(customBackendName)
        else:
            Logger.error("Unknown quantum backend specified!")
        return backend


class VQEMaxCut(Clustering):
    def __init__(self,
                 number_of_clusters = 1,
                 max_trials: int = 1,
                 reps: int = 1,
                 entanglement: str = 'linear',
                 backend = QuantumBackends.aer_statevector_simulator,
                 ibmq_token: str = "",
                 ibmq_custom_backend: str = ""):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        self.__max_trials = max_trials
        self.__reps = reps
        self.__entanglement = entanglement
        self.__backend = backend
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__vqeAlgorithmus(similarity_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(similarity_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(similarity_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(self.__number_of_clusters, similarity_matrix, label , label_all , 1)
            #print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(self, iteration : int, similarity_matrix: np.matrix , label: np.matrix , label_all: np.matrix , category: int) -> np.matrix:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label
            new_label = self.__vqeAlgorithmus(similarity_matrix)
        
            z = -1
            check_label = np.ones(len(label))
            check_label.astype(np.int)
            for i in range(len(label)):
                check_label[i] = label[i]
            for i in range(len(label)):
                if check_label[i] == category:
                    z = z+1
                    label_all[i] = label_all[i] + new_label[z]*pow(2,iteration-1)
                    label[i] = new_label[z]
            Logger.normal("label after "+str(iteration) + " iteration :" + str(label_all))

            # ones: rekursion only with ones labels in new label
            ones = self.__split_Matrix(similarity_matrix,new_label,1)
            self.__rekursivAlgorithmus(iteration-1, ones, label , label_all , 1)
            
            # change label for the zero cluster
            z = -1
            for i in range(len(label)):
                if check_label[i] == 1:
                    z = z+1
                    if new_label[z] == 0:
                        label[i] = 1
                    else: 
                        label[i] = 0
                else : 
                    label[i] = 0
            
            #zeros: rekursion only with zero labels in new label
            zeros = self.__split_Matrix(similarity_matrix,new_label,0)

            self.__rekursivAlgorithmus(iteration-1, zeros, label , label_all , 1)   
            return label_all

    def __split_Matrix(self, similarity_matrix : np.matrix , label : np.matrix , category : int) -> np.matrix:
        # split the similarity matrix in one smaller matrix. These matrix contains only similarities with the right label
        npl = 0
        for i in range(len(label)):
            if label[i] == category:
                npl = npl+1

        NSM = np.zeros((npl,npl))
        s = -1
        t = -1
        for i in range(len(label)):
            if label[i] == category:
                s += 1
                t = -1
                for j in range(len(label)):
                    if label[j] == category:
                        t += 1
                        NSM[s,t] = similarity_matrix[i,j]
        return NSM

    def __vqeAlgorithmus(self, similarity_matrix: np.matrix) -> np.matrix:
        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            label = np.zeros(similarity_matrix.shape[0])
            return label.astype(np.int)
        qubitOp, offset = max_cut.get_operator(similarity_matrix)
        seed = 10598

        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

        spsa = SPSA(max_trials=self.__max_trials)
        ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=self.__reps, entanglement=self.__entanglement)
        vqe = VQE(qubitOp, ry, spsa, quantum_instance=quantum_instance)

        # run VQE
        result = vqe.run(quantum_instance)

        # print results
        x = sample_most_likely(result.eigenstate)
        print('energy:', result.eigenvalue.real)
        print('time:', result.optimizer_time)
        print('max-cut objective:', result.eigenvalue.real + offset)
        print('solution:', max_cut.get_graph_solution(x))
        print('solution objective:', max_cut.max_cut_value(x, similarity_matrix))
        solution = max_cut.get_graph_solution(x)
        return solution.astype(np.int) 

    # getter and setter methodes
    def get_number_of_clusters(self) -> int:
        return self.__number_of_clusters
    def get_max_trials(self) -> int:
        return self.__max_trials
    def get_reps(self) -> int:
        return self.__reps
    def get_entanglement(self) -> str:
        return self.__entanglement
    def get_backend(self) -> str:
        return self.__backend
    def get_ibmq_token(self) -> str:
        return self.__ibmq_token
    def get_ibmq_custom_backend(self):
        return self.__ibmq_custom_backend

    def set_number_of_clusters(self, number_of_clusters : int = 1) -> None:
        if isinstance(number_of_clusters, int) and number_of_clusters > 0:
            self.__number_of_clusters = number_of_clusters
    def set_max_trials(self, max_trials : int = 1) -> None:
        if isinstance(max_trials, int) and max_trials > 0:
            self.__max_trials = max_trials
    def set_reps(self, reps = 1 ) -> int:
        if isinstance(reps, int) and reps > 0:
            self.__reps = reps
    def set_entanglement(self, entanglement : str = 'linear') -> str:
        self.__entanglement = entanglement
    def set_backend(self, backend: str = QuantumBackends.aer_statevector_simulator) -> None:
        self.__backend = backend
    def set_ibmq_token(self, ibmq_token: str = "") -> None:
        self.__ibmq_token = ibmq_token
    def set_ibmq_custom_backend(self, ibmq_custom_backend: str = ""):
        self.__ibmq_custom_backend = ibmq_custom_backend

    #getter and setter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "QAOA MaxCut"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=1)"\
                            +"2**x Clusters would be generated"
        params.append(("numberClusters", "Number of Clusters" ,description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))
        
        parameter_max_trials = self.get_max_trials()
        description_max_trials = "int > 0 (default 1) "\
                            +"For Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer:"\
                            +"Maximum number of iterations to perform."
        params.append(("maxTrials", "max Trials" ,description_max_trials, parameter_max_trials,"number", 1 , 1 ))
        
        parameter_reps = self.get_reps()
        description_reps = "int > 0 (default 1) "\
                            +"For The two-local circuit:"\
                            +"Specifies how often a block consisting of a rotation layer and entanglement layer is repeated."
        params.append(("reps" , "Reps" ,description_reps, parameter_reps, "number" , 1,1 ))

        parameter_entanglement = self.get_entanglement()
        description_entanglement = "str default('linear') "\
                            +"A set of default entanglement strategies is provided:"\
                            +"'full' entanglement is each qubit is entangled with all the others."\
                            +"'linear' entanglement is qubit i entangled with qubit i+1, for all i∈{0,1,...,n−2}, where n is the total number of qubits."\
                            +"'circular' entanglement is linear entanglement but with an additional entanglement of the first and last qubit before the linear part."\
                            +"'sca' (shifted-circular-alternating) entanglement is a generalized and modified version of the proposed circuit 14 in Sim et al.. "\
                            +"It consists of circular entanglement where the ‘long’ entanglement connecting the first with the last qubit is shifted by one each block."\
                            +"Furthermore the role of control and target qubits are swapped every block (therefore alternating)."

        params.append(("entanglement", "Entanglement" ,description_entanglement, parameter_entanglement , "select" , ('full','linear','circlular', 'sca')))

        parameter_backend = self.get_backend().value
        description_backend = "Enum default(aer_statevector_simulator) "\
            + " A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."\
            + " When using (custom_ibmq), a custom ibmq backend can be specified."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmq_token = self.get_ibmq_token()
        description_ibmq_token = "str default(\"\") "\
            + " The token of an account accessing the IBMQ online service."
        params.append(("ibmqToken", "IBMQ-Token", description_ibmq_token, parameter_ibmq_token, "text", "", ""))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxTrials":
                self.set_max_trials(param[3])
            elif param[0] == "reps":
                self.set_reps(param[3])
            elif param[0] == "entanglement":
                self.set_entanglement(param[3])
            elif param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            elif param[0] == "ibmqToken":
                self.set_ibmq_token(param[3])
            elif param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class ClassicNaiveMaxCut(Clustering):
    def __init__(self, number_of_clusters = 1):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__classicNaiveMaxCutAlgo(similarity_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(similarity_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(similarity_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(self.__number_of_clusters, similarity_matrix, label , label_all , 1)
            #print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(self, iteration : int, similarity_matrix: np.matrix , label: np.matrix , label_all: np.matrix , category: int) -> np.matrix:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label
            new_label = self.__classicNaiveMaxCutAlgo(similarity_matrix)
        
            z = -1
            check_label = np.ones(len(label))
            check_label.astype(np.int)
            for i in range(len(label)):
                check_label[i] = label[i]
            for i in range(len(label)):
                if check_label[i] == category:
                    z = z+1
                    label_all[i] = label_all[i] + new_label[z]*pow(2,iteration-1)
                    label[i] = new_label[z]
            Logger.normal("label after "+str(iteration) + " iteration :" + str(label_all))

            # ones: rekursion only with ones labels in new label
            ones = self.__split_Matrix(similarity_matrix,new_label,1)
            self.__rekursivAlgorithmus(iteration-1, ones, label , label_all , 1)
            
            # change label for the zero cluster
            z = -1
            for i in range(len(label)):
                if check_label[i] == 1:
                    z = z+1
                    if new_label[z] == 0:
                        label[i] = 1
                    else: 
                        label[i] = 0
                else : 
                    label[i] = 0
            
            #zeros: rekursion only with zero labels in new label
            zeros = self.__split_Matrix(similarity_matrix,new_label,0)

            self.__rekursivAlgorithmus(iteration-1, zeros, label , label_all , 1)   
            return label_all

    def __split_Matrix(self, similarity_matrix : np.matrix , label : np.matrix , category : int) -> np.matrix:
        # split the similarity matrix in one smaller matrix. These matrix contains only similarities with the right label
        npl = 0
        for i in range(len(label)):
            if label[i] == category:
                npl = npl+1

        NSM = np.zeros((npl,npl))
        s = -1
        t = -1
        for i in range(len(label)):
            if label[i] == category:
                s += 1
                t = -1
                for j in range(len(label)):
                    if label[j] == category:
                        t += 1
                        NSM[s,t] = similarity_matrix[i,j]
        return NSM

    def __create_graph(self, similarity_matrix: np.matrix) -> nx.Graph:
        probSize = similarity_matrix.shape[0]
        graph = nx.Graph()

        for i in range(0, probSize):
            for j in range(0, probSize):
                if i != j:
                    graph.add_edge(i, j, weight = similarity_matrix[i][j])
        
        return graph

    def __classicNaiveMaxCutAlgo(self, similarity_matrix: np.matrix) -> np.matrix:
        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # Create classic naive max cut solver
        graph = self.__create_graph(similarity_matrix)

        # Solve

        check: Timer = Timer()    
        check.start()

        solver = ClassicNaiveMaxCutSolver(graph)
        (cutValue, cutEdges) = solver.solve()

        # Remove the max cut edges
        for (u, v) in cutEdges:
            graph.remove_edge(u, v)

        # Plot the graphs
        #from matplotlib import pyplot as plt
        #pos = nx.spring_layout(graph)
        #nx.draw(graph, pos)
        #labels = nx.get_edge_attributes(graph, 'weight')
        #nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
        #plt.savefig("CuttedGraph.png", format="PNG")
        #plt.clf()

        # define element 0 (left side of first cut) is cluster 0
        element0 = cutEdges[0][0]
        label[element0] = 0

        for node in graph.nodes():
            # if node has path to element 0, then cluster 0
            # if not then cluster 1
            if nx.algorithms.shortest_paths.generic.has_path(graph, element0, node):
                label[node] = 0
            else:
                label[node] = 1

        check.stop()

        # print results
        print('solution:', str(cutEdges))
        print('solution objective:', str(cutValue))

        return label.astype(np.int)

    # getter and setter methodes
    def get_number_of_clusters(self) -> int:
        return self.__number_of_clusters

    def set_number_of_clusters(self, number_of_clusters : int = 2) -> None:
        if isinstance(number_of_clusters, int) and number_of_clusters > 0:
            self.__number_of_clusters = number_of_clusters

    #getter and setter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Classic Naive MaxCut"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +"2**x Clusters would be generated"
        params.append(("numberClusters", "Number of Clusters" ,description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])
    
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class SdpMaxCut(Clustering):
    def __init__(self, number_of_clusters = 1):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__sdpMaxCutAlgo(similarity_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(similarity_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(similarity_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(self.__number_of_clusters, similarity_matrix, label , label_all , 1)
            #print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(self, iteration : int, similarity_matrix: np.matrix , label: np.matrix , label_all: np.matrix , category: int) -> np.matrix:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label
            new_label = self.__sdpMaxCutAlgo(similarity_matrix)
        
            z = -1
            check_label = np.ones(len(label))
            check_label.astype(np.int)
            for i in range(len(label)):
                check_label[i] = label[i]
            for i in range(len(label)):
                if check_label[i] == category:
                    z = z+1
                    label_all[i] = label_all[i] + new_label[z]*pow(2,iteration-1)
                    label[i] = new_label[z]
            Logger.normal("label after "+str(iteration) + " iteration :" + str(label_all))

            # ones: rekursion only with ones labels in new label
            ones = self.__split_Matrix(similarity_matrix,new_label,1)
            self.__rekursivAlgorithmus(iteration-1, ones, label , label_all , 1)
            
            # change label for the zero cluster
            z = -1
            for i in range(len(label)):
                if check_label[i] == 1:
                    z = z+1
                    if new_label[z] == 0:
                        label[i] = 1
                    else: 
                        label[i] = 0
                else : 
                    label[i] = 0
            
            #zeros: rekursion only with zero labels in new label
            zeros = self.__split_Matrix(similarity_matrix,new_label,0)

            self.__rekursivAlgorithmus(iteration-1, zeros, label , label_all , 1)   
            return label_all

    def __split_Matrix(self, similarity_matrix : np.matrix , label : np.matrix , category : int) -> np.matrix:
        # split the similarity matrix in one smaller matrix. These matrix contains only similarities with the right label
        npl = 0
        for i in range(len(label)):
            if label[i] == category:
                npl = npl+1

        NSM = np.zeros((npl,npl))
        s = -1
        t = -1
        for i in range(len(label)):
            if label[i] == category:
                s += 1
                t = -1
                for j in range(len(label)):
                    if label[j] == category:
                        t += 1
                        NSM[s,t] = similarity_matrix[i,j]
        return NSM

    def __create_graph(self, similarity_matrix: np.matrix) -> nx.Graph:
        probSize = similarity_matrix.shape[0]
        graph = nx.Graph()

        for i in range(0, probSize):
            for j in range(0, probSize):
                if i != j:
                    graph.add_edge(i, j, weight = similarity_matrix[i][j])
        
        return graph

    def __sdpMaxCutAlgo(self, similarity_matrix: np.matrix) -> np.matrix:
        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # Create sdp max cut solver
        graph = self.__create_graph(similarity_matrix)

        # Solve

        check: Timer = Timer()    
        check.start()

        solver = SdpMaxCutSolver(graph)
        (cutValue, cutEdges) = solver.solve()

        # Remove the max cut edges
        for (u, v) in cutEdges:
            graph.remove_edge(u, v)

        # Plot the graphs
        #from matplotlib import pyplot as plt
        #pos = nx.spring_layout(graph)
        #nx.draw(graph, pos)
        #labels = nx.get_edge_attributes(graph, 'weight')
        #nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
        #plt.savefig("CuttedGraph.png", format="PNG")
        #plt.clf()

        # define element 0 (left side of first cut) is cluster 0
        element0 = cutEdges[0][0]
        label[element0] = 0

        for node in graph.nodes():
            # if node has path to element 0, then cluster 0
            # if not then cluster 1
            if nx.algorithms.shortest_paths.generic.has_path(graph, element0, node):
                label[node] = 0
            else:
                label[node] = 1

        check.stop()

        # print results
        print('solution:', str(cutEdges))
        print('solution objective:', str(cutValue))

        return label.astype(np.int)

    # getter and setter methodes
    def get_number_of_clusters(self) -> int:
        return self.__number_of_clusters

    def set_number_of_clusters(self, number_of_clusters : int = 2) -> None:
        if isinstance(number_of_clusters, int) and number_of_clusters > 0:
            self.__number_of_clusters = number_of_clusters

    #getter and setter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Semidefinite Programming MaxCut"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +"2**x Clusters would be generated"
        params.append(("numberClusters", "Number of Clusters" ,description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])
    
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class BmMaxCut(Clustering):
    def __init__(self, number_of_clusters = 1):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__bmMaxCutAlgo(similarity_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(similarity_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(similarity_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(self.__number_of_clusters, similarity_matrix, label , label_all , 1)
            #print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(self, iteration : int, similarity_matrix: np.matrix , label: np.matrix , label_all: np.matrix , category: int) -> np.matrix:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label
            new_label = self.__bmMaxCutAlgo(similarity_matrix)
        
            z = -1
            check_label = np.ones(len(label))
            check_label.astype(np.int)
            for i in range(len(label)):
                check_label[i] = label[i]
            for i in range(len(label)):
                if check_label[i] == category:
                    z = z+1
                    label_all[i] = label_all[i] + new_label[z]*pow(2,iteration-1)
                    label[i] = new_label[z]
            Logger.normal("label after "+str(iteration) + " iteration :" + str(label_all))

            # ones: rekursion only with ones labels in new label
            ones = self.__split_Matrix(similarity_matrix,new_label,1)
            self.__rekursivAlgorithmus(iteration-1, ones, label , label_all , 1)
            
            # change label for the zero cluster
            z = -1
            for i in range(len(label)):
                if check_label[i] == 1:
                    z = z+1
                    if new_label[z] == 0:
                        label[i] = 1
                    else: 
                        label[i] = 0
                else : 
                    label[i] = 0
            
            #zeros: rekursion only with zero labels in new label
            zeros = self.__split_Matrix(similarity_matrix,new_label,0)

            self.__rekursivAlgorithmus(iteration-1, zeros, label , label_all , 1)   
            return label_all

    def __split_Matrix(self, similarity_matrix : np.matrix , label : np.matrix , category : int) -> np.matrix:
        # split the similarity matrix in one smaller matrix. These matrix contains only similarities with the right label
        npl = 0
        for i in range(len(label)):
            if label[i] == category:
                npl = npl+1

        NSM = np.zeros((npl,npl))
        s = -1
        t = -1
        for i in range(len(label)):
            if label[i] == category:
                s += 1
                t = -1
                for j in range(len(label)):
                    if label[j] == category:
                        t += 1
                        NSM[s,t] = similarity_matrix[i,j]
        return NSM

    def __create_graph(self, similarity_matrix: np.matrix) -> nx.Graph:
        probSize = similarity_matrix.shape[0]
        graph = nx.Graph()

        for i in range(0, probSize):
            for j in range(0, probSize):
                if i != j:
                    graph.add_edge(i, j, weight = similarity_matrix[i][j])
        
        return graph

    def __bmMaxCutAlgo(self, similarity_matrix: np.matrix) -> np.matrix:
        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # Create sdp max cut solver
        graph = self.__create_graph(similarity_matrix)

        # Solve

        check: Timer = Timer()    
        check.start()

        solver = BmMaxCutSolver(graph)
        (cutValue, cutEdges) = solver.solve()

        # Remove the max cut edges
        for (u, v) in cutEdges:
            graph.remove_edge(u, v)

        # Plot the graphs
        #from matplotlib import pyplot as plt
        #pos = nx.spring_layout(graph)
        #nx.draw(graph, pos)
        #labels = nx.get_edge_attributes(graph, 'weight')
        #nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
        #plt.savefig("CuttedGraph.png", format="PNG")
        #plt.clf()

        # define element 0 (left side of first cut) is cluster 0
        element0 = cutEdges[0][0]
        label[element0] = 0

        for node in graph.nodes():
            # if node has path to element 0, then cluster 0
            # if not then cluster 1
            if nx.algorithms.shortest_paths.generic.has_path(graph, element0, node):
                label[node] = 0
            else:
                label[node] = 1

        check.stop()

        # print results
        print('solution:', str(cutEdges))
        print('solution objective:', str(cutValue))

        return label.astype(np.int)

    # getter and setter methodes
    def get_number_of_clusters(self) -> int:
        return self.__number_of_clusters

    def set_number_of_clusters(self, number_of_clusters : int = 2) -> None:
        if isinstance(number_of_clusters, int) and number_of_clusters > 0:
            self.__number_of_clusters = number_of_clusters

    #getter and setter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Bureir-Monteiro solver for MaxCut"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +"2**x Clusters would be generated"
        params.append(("numberClusters", "Number of Clusters" ,description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])
    
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class NegativeRotationQuantumKMeansClustering(Clustering):
    def __init__(self, 
        number_of_clusters = 2,
        max_qubits = 2,
        shots_each = 100,
        max_runs = 10,
        relative_residual_amount = 5,
        backend = QuantumBackends.aer_statevector_simulator,
        ibmq_token = "",
        ibmq_custom_backend = ""):
        super().__init__()
        self.clusterAlgo = NegativeRotationQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        self.clusterAlgo.set_number_of_clusters(self.get_number_of_clusters())
        self.clusterAlgo.set_max_qubits(self.get_max_qubits())
        self.clusterAlgo.set_shots_each(self.get_shots_each())
        self.clusterAlgo.set_max_runs(self.get_max_runs())
        self.clusterAlgo.set_relative_residual_amount(self.get_relative_residual_amount())

        qBackend = QuantumBackends.get_quantum_backend(self.backend, self.ibmq_token, self.ibmq_custom_backend)

        self.clusterAlgo.set_backend(qBackend)

        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # run
        clusterMapping = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int)

    #getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_max_qubits(self):
        return self.max_qubits

    def set_max_qubits(self, max_qubits):
        self.max_qubits = max_qubits
        return

    def get_shots_each(self):
        return self.shots_each

    def set_shots_each(self, shots_each):
        self.shots_each = shots_each
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def get_relative_residual_amount(self):
        return self.relative_residual_amount

    def set_relative_residual_amount(self, relative_residual_amount):
        self.relative_residual_amount = relative_residual_amount
        return

    def get_backend(self):
        return self.backend

    def set_backend(self, backend):
        self.backend = backend
        return

    def get_ibmq_token(self):
        return self.ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.ibmq_token = ibmq_token
        return

    def get_ibmq_custom_backend(self):
        return self.ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Negative Rotation Quantum KMeans"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters" , description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))
        
        parameter_max_qubits = self.get_max_qubits()
        description_max_qubits = "int > 0 (default 2): "\
                            +"The amount of qubits that are used for executing the circuits."
        params.append(("maxQubits", "max qubits" ,description_max_qubits,parameter_max_qubits,"number", 1 , 1 ))

        parameter_shots_each = self.get_shots_each()
        description_shots_each = "int > 0 (default 100): "\
                            +"The amount of shots for each circuit run."
        params.append(("shotsEach", "shots each" ,description_shots_each,parameter_shots_each,"number", 1 , 1 ))

        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): "\
                            +"The amount of k mean iteration runs."
        params.append(("maxRuns", "max runs" ,description_max_runs,parameter_max_runs,"number", 1 , 1 ))

        parameter_relative_residual = self.get_relative_residual_amount()
        description_relative_residual = "int > 0 (default 5): "\
                            +"The amount in percentage of how many data points can change their label between" \
                            + "two runs. The default is 5, i.e. when less then 5% of the data points change" \
                            + "their label, we consider this as converged"
        params.append(("relativeResidual", "relative residual amount" ,description_relative_residual,parameter_relative_residual,"number", 1 , 1 ))

        parameter_backend = self.get_backend().value
        description_backend = "Enum default(aer_statevector_simulator): "\
            + " A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmq_token = self.get_ibmq_token()
        description_ibmq_token = "str default(\"\") "\
            + " The token of an account accessing the IBMQ online service."
        params.append(("ibmqToken", "IBMQ-Token", description_ibmq_token, parameter_ibmq_token, "text", "", ""))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxQubits":
                self.set_max_qubits(param[3])
            elif param[0] == "shotsEach":
                self.set_shots_each(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "relativeResidual":
                self.set_relative_residual_amount(param[3])
            elif param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            elif param[0] == "ibmqToken":
                self.set_ibmq_token(param[3])
            elif param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class DestructiveInterferenceQuantumKMeansClustering(Clustering):
    def __init__(self, 
        number_of_clusters = 2,
        max_qubits = 2,
        shots_each = 100,
        max_runs = 10,
        relative_residual_amount = 5,
        backend = QuantumBackends.aer_statevector_simulator,
        ibmq_token = "",
        ibmq_custom_backend = ""):
        super().__init__()
        self.clusterAlgo = DestructiveInterferenceQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        self.clusterAlgo.set_number_of_clusters(self.get_number_of_clusters())
        self.clusterAlgo.set_max_qubits(self.get_max_qubits())
        self.clusterAlgo.set_shots_each(self.get_shots_each())
        self.clusterAlgo.set_max_runs(self.get_max_runs())
        self.clusterAlgo.set_relative_residual_amount(self.get_relative_residual_amount())

        qBackend = QuantumBackends.get_quantum_backend(self.backend, self.ibmq_token, self.ibmq_custom_backend)

        self.clusterAlgo.set_backend(qBackend)

        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # run
        clusterMapping = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int)

    #getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_max_qubits(self):
        return self.max_qubits

    def set_max_qubits(self, max_qubits):
        self.max_qubits = max_qubits
        return

    def get_shots_each(self):
        return self.shots_each

    def set_shots_each(self, shots_each):
        self.shots_each = shots_each
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def get_relative_residual_amount(self):
        return self.relative_residual_amount

    def set_relative_residual_amount(self, relative_residual_amount):
        self.relative_residual_amount = relative_residual_amount
        return

    def get_backend(self):
        return self.backend

    def set_backend(self, backend):
        self.backend = backend
        return

    def get_ibmq_token(self):
        return self.ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.ibmq_token = ibmq_token
        return

    def get_ibmq_custom_backend(self):
        return self.ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Destructive Interference Quantum KMeans"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters" , description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))
        
        parameter_max_qubits = self.get_max_qubits()
        description_max_qubits = "int > 0 (default 2): "\
                            +"The amount of qubits that are used for executing the circuits."
        params.append(("maxQubits", "max qubits" ,description_max_qubits,parameter_max_qubits,"number", 1 , 1 ))

        parameter_shots_each = self.get_shots_each()
        description_shots_each = "int > 0 (default 100): "\
                            +"The amount of shots for each circuit run."
        params.append(("shotsEach", "shots each" ,description_shots_each,parameter_shots_each,"number", 1 , 1 ))

        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): "\
                            +"The amount of k mean iteration runs."
        params.append(("maxRuns", "max runs" ,description_max_runs,parameter_max_runs,"number", 1 , 1 ))

        parameter_relative_residual = self.get_relative_residual_amount()
        description_relative_residual = "int > 0 (default 5): "\
                            +"The amount in percentage of how many data points can change their label between" \
                            + "two runs. The default is 5, i.e. when less then 5% of the data points change" \
                            + "their label, we consider this as converged"
        params.append(("relativeResidual", "relative residual amount" ,description_relative_residual,parameter_relative_residual,"number", 1 , 1 ))

        parameter_backend = self.get_backend().value
        description_backend = "Enum default(aer_statevector_simulator): "\
            + " A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmq_token = self.get_ibmq_token()
        description_ibmq_token = "str default(\"\") "\
            + " The token of an account accessing the IBMQ online service."
        params.append(("ibmqToken", "IBMQ-Token", description_ibmq_token, parameter_ibmq_token, "text", "", ""))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxQubits":
                self.set_max_qubits(param[3])
            elif param[0] == "shotsEach":
                self.set_shots_each(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "relativeResidual":
                self.set_relative_residual_amount(param[3])
            elif param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            elif param[0] == "ibmqToken":
                self.set_ibmq_token(param[3])
            elif param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class StatePreparationQuantumKMeansClustering(Clustering):
    def __init__(self, 
        number_of_clusters = 2,
        max_qubits = 2,
        shots_each = 100,
        max_runs = 10,
        relative_residual_amount = 5,
        backend = QuantumBackends.aer_statevector_simulator,
        ibmq_token = "",
        ibmq_custom_backend = ""):
        super().__init__()
        self.clusterAlgo = StatePreparationQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return
        
    def create_cluster(self, position_matrix : np.matrix , similarity_matrix : np.matrix) -> np.matrix:
        self.clusterAlgo.set_number_of_clusters(self.get_number_of_clusters())
        self.clusterAlgo.set_max_qubits(self.get_max_qubits())
        self.clusterAlgo.set_shots_each(self.get_shots_each())
        self.clusterAlgo.set_max_runs(self.get_max_runs())
        self.clusterAlgo.set_relative_residual_amount(self.get_relative_residual_amount())

        qBackend = QuantumBackends.get_quantum_backend(self.backend, self.ibmq_token, self.ibmq_custom_backend)

        self.clusterAlgo.set_backend(qBackend)

        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # run
        clusterMapping = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int)

    #getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_max_qubits(self):
        return self.max_qubits

    def set_max_qubits(self, max_qubits):
        self.max_qubits = max_qubits
        return

    def get_shots_each(self):
        return self.shots_each

    def set_shots_each(self, shots_each):
        self.shots_each = shots_each
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def get_relative_residual_amount(self):
        return self.relative_residual_amount

    def set_relative_residual_amount(self, relative_residual_amount):
        self.relative_residual_amount = relative_residual_amount
        return

    def get_backend(self):
        return self.backend

    def set_backend(self, backend):
        self.backend = backend
        return

    def get_ibmq_token(self):
        return self.ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.ibmq_token = ibmq_token
        return

    def get_ibmq_custom_backend(self):
        return self.ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "State Preparation Quantum KMeans"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters" , description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))
        
        parameter_max_qubits = self.get_max_qubits()
        description_max_qubits = "int > 0 (default 2): "\
                            +"The amount of qubits that are used for executing the circuits."
        params.append(("maxQubits", "max qubits" ,description_max_qubits,parameter_max_qubits,"number", 1 , 1 ))

        parameter_shots_each = self.get_shots_each()
        description_shots_each = "int > 0 (default 100): "\
                            +"The amount of shots for each circuit run."
        params.append(("shotsEach", "shots each" ,description_shots_each,parameter_shots_each,"number", 1 , 1 ))

        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): "\
                            +"The amount of k mean iteration runs."
        params.append(("maxRuns", "max runs" ,description_max_runs,parameter_max_runs,"number", 1 , 1 ))

        parameter_relative_residual = self.get_relative_residual_amount()
        description_relative_residual = "int > 0 (default 5): "\
                            +"The amount in percentage of how many data points can change their label between" \
                            + "two runs. The default is 5, i.e. when less then 5% of the data points change" \
                            + "their label, we consider this as converged"
        params.append(("relativeResidual", "relative residual amount" ,description_relative_residual,parameter_relative_residual,"number", 1 , 1 ))

        parameter_backend = self.get_backend().value
        description_backend = "Enum default(aer_statevector_simulator): "\
            + " A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmq_token = self.get_ibmq_token()
        description_ibmq_token = "str default(\"\") "\
            + " The token of an account accessing the IBMQ online service."
        params.append(("ibmqToken", "IBMQ-Token", description_ibmq_token, parameter_ibmq_token, "text", "", ""))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxQubits":
                self.set_max_qubits(param[3])
            elif param[0] == "shotsEach":
                self.set_shots_each(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "relativeResidual":
                self.set_relative_residual_amount(param[3])
            elif param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            elif param[0] == "ibmqToken":
                self.set_ibmq_token(param[3])
            elif param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class PositiveCorrelationQuantumKMeansClustering(Clustering):
    def __init__(self,
                 number_of_clusters=2,
                 shots_each=100,
                 max_runs=10,
                 relative_residual_amount=5,
                 backend=QuantumBackends.aer_statevector_simulator,
                 ibmq_token="",
                 ibmq_custom_backend=""):
        super().__init__()
        self.clusterAlgo = PositiveCorrelationQuantumKmeans()

        self.number_of_clusters = number_of_clusters
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def create_cluster(self, position_matrix: np.matrix, similarity_matrix: np.matrix) -> np.matrix:
        qBackend = QuantumBackends.get_quantum_backend(self.backend, self.ibmq_token, self.ibmq_custom_backend)

        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # run
        clusterMapping = self.clusterAlgo.fit(position_matrix, self.number_of_clusters, self.max_runs, self.relative_residual_amount, qBackend, self.shots_each)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int)

    # getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_shots_each(self):
        return self.shots_each

    def set_shots_each(self, shots_each):
        self.shots_each = shots_each
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def get_relative_residual_amount(self):
        return self.relative_residual_amount

    def set_relative_residual_amount(self, relative_residual_amount):
        self.relative_residual_amount = relative_residual_amount
        return

    def get_backend(self):
        return self.backend

    def set_backend(self, backend):
        self.backend = backend
        return

    def get_ibmq_token(self):
        return self.ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.ibmq_token = ibmq_token
        return

    def get_ibmq_custom_backend(self):
        return self.ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Positive Correlation Quantum KMeans"
        params.append(("name", "ClusterTyp", "Name of choosen Clustering Type", clusteringTypeName, "header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)" \
                                         + ": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters", description_number_of_clusters,
                       parameter_number_of_clusters, "number", 1, 1))

        parameter_shots_each = self.get_shots_each()
        description_shots_each = "int > 0 (default 100): " \
                                 + "The amount of shots for each circuit run."
        params.append(("shotsEach", "shots each", description_shots_each, parameter_shots_each, "number", 1, 1))

        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): " \
                               + "The amount of k mean iteration runs."
        params.append(("maxRuns", "max runs", description_max_runs, parameter_max_runs, "number", 1, 1))

        parameter_relative_residual = self.get_relative_residual_amount()
        description_relative_residual = "int > 0 (default 5): " \
                                        + "The amount in percentage of how many data points can change their label between" \
                                        + "two runs. The default is 5, i.e. when less then 5% of the data points change" \
                                        + "their label, we consider this as converged"
        params.append(("relativeResidual", "relative residual amount", description_relative_residual,
                       parameter_relative_residual, "number", 1, 1))

        parameter_backend = self.get_backend().value
        description_backend = "Enum default(aer_statevector_simulator): " \
                              + " A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select",
                       [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") " \
                                          + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend,
                       str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmq_token = self.get_ibmq_token()
        description_ibmq_token = "str default(\"\") " \
                                 + " The token of an account accessing the IBMQ online service."
        params.append(("ibmqToken", "IBMQ-Token", description_ibmq_token, parameter_ibmq_token, "text", "", ""))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params

    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "shotsEach":
                self.set_shots_each(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "relativeResidual":
                self.set_relative_residual_amount(param[3])
            elif param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            elif param[0] == "ibmqToken":
                self.set_ibmq_token(param[3])
            elif param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class ClassicalKMeans(Clustering):
    def __init__(
        self,
        number_of_clusters = 2,
        max_runs = 10,
        relative_residual_amount = 5
    ):
        super().__init__()
        self.number_of_clusters = number_of_clusters
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        return

    def create_cluster(self, position_matrix : np.matrix, similarity_matrix : np.matrix ) -> np.matrix:
        n_clusters = self.get_number_of_clusters()
        random_state = 0
        max_iter = self.get_max_runs()
        tol = self.get_relative_residual_amount() / 100.0
        kmeansOutput = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            max_iter=max_iter,
            tol=tol).fit(position_matrix)
        return kmeansOutput.labels_.astype(np.int)

    #getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def get_relative_residual_amount(self):
        return self.relative_residual_amount

    def set_relative_residual_amount(self, relative_residual_amount):
        self.relative_residual_amount = relative_residual_amount
        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Classical KMeans"
        params.append(("name", "ClusterTyp" ,"Name of choosen Clustering Type", clusteringTypeName ,"header"))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)"\
                            +": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters" , description_number_of_clusters, parameter_number_of_clusters, "number", 1 , 1 ))
        
        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): "\
                            +"The amount of k mean iteration runs."
        params.append(("maxRuns", "max runs" ,description_max_runs,parameter_max_runs,"number", 1 , 1 ))

        parameter_relative_residual = self.get_relative_residual_amount()
        description_relative_residual = "int > 0 (default 5): "\
                            +"The amount in percentage of how many data points can change their label between" \
                            + "two runs. The default is 5, i.e. when less then 5% of the data points change" \
                            + "their label, we consider this as converged"
        params.append(("relativeResidual", "relative residual amount" ,description_relative_residual,parameter_relative_residual,"number", 1 , 1 ))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                                           + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping,
                       parameter_keep_cluster_mapping, "checkbox"))

        return params
        
    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "relativeResidual":
                self.set_relative_residual_amount(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume] ) -> None:
        pass


class ClassicalKMedoids(Clustering):
    def __init__(
            self,
            number_of_clusters=2,
            max_runs=10
    ):
        super().__init__()
        self.number_of_clusters = number_of_clusters
        self.max_runs = max_runs
        self.method = 'alternate'
        self.init = 'build'
        return

    def create_cluster(self, position_matrix: np.matrix, similarity_matrix: np.matrix) -> np.matrix:
        n_clusters = self.get_number_of_clusters()
        random_state = 0
        method = self.get_method()
        init = self.get_init()
        max_iter = self.get_max_runs()
        kmedoidsOutput = KMedoids(
            n_clusters=n_clusters,
            method=method,
            init=init,
            random_state=random_state,
            max_iter=max_iter).fit(similarity_matrix)
        return kmedoidsOutput.labels_.astype(np.int)

    # getter and setter params
    def get_number_of_clusters(self):
        return self.number_of_clusters

    def set_number_of_clusters(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters
        return

    def get_max_runs(self):
        return self.max_runs

    def set_max_runs(self, max_runs):
        self.max_runs = max_runs
        return

    def set_method(self, method):
        self.method = method
        return

    def get_method(self):
        return self.method

    def set_init(self, init):
        self.init = init
        return

    def get_init(self):
        return self.init

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        clusteringTypeName = "Classical KMedoids"
        params.append(("name", "ClusterTyp", "Name of choosen Clustering Type", clusteringTypeName, "header"))

        parameter_init = self.get_init()
        description_init = "string (default=build)" \
                                         + "Specify medoid initialization method. ‘random’ selects n_clusters " \
                                           "elements from the dataset. ‘heuristic’ picks the n_clusters points with the " \
                                           "smallest sum distance to every other point. ‘k-medoids++’ follows an " \
                                           "approach based on k-means++_, and in general, gives initial medoids which " \
                                         "are more separated than those generated by the other methods. ‘build’ is a " \
                                         "greedy initialization of the medoids used in the original PAM algorithm. " \
                                         "Often ‘build’ is more efficient but slower than other initializations on " \
                                         "big datasets and it is also very non-robust, if there are outliers in the " \
                                         "dataset, use another initialization. "
        params.append(("init", "Initialization", description_init,
                       parameter_init, "text", "", ""))

        parameter_method = self.get_method()
        description_method = "string (default=alternate)" \
                                         + "Which algorithm to use. ‘alternate’ is faster while ‘pam’ is more accurate."
        params.append(("method", "Method", description_method,
                       parameter_method, "text", "", ""))

        parameter_number_of_clusters = self.get_number_of_clusters()
        description_number_of_clusters = "int > 0 (default=2)" \
                                         + ": k Clusters will be generated"
        params.append(("numberClusters", "Number of Clusters", description_number_of_clusters,
                       parameter_number_of_clusters, "number", 1, 1))

        parameter_max_runs = self.get_max_runs()
        description_max_runs = "int > 0 (default 10): " \
                               + "The amount of k medoids iteration runs."
        params.append(("maxRuns", "max runs", description_max_runs, parameter_max_runs, "number", 1, 1))

        parameter_keep_cluster_mapping = self.get_keep_cluster_mapping()
        description_keep_cluster_mapping = "bool (default False): " \
                               + "If True, keeps the cluster mapping when re-calculating."
        params.append(("keepClusterMapping", "keep cluster mapping", description_keep_cluster_mapping, parameter_keep_cluster_mapping, "checkbox"))

        return params

    def set_param_list(self, params: list = []) -> np.matrix:
        for param in params:
            if param[0] == "numberClusters":
                self.set_number_of_clusters(param[3])
            elif param[0] == "maxRuns":
                self.set_max_runs(param[3])
            elif param[0] == "init":
                self.set_init(param[3])
            elif param[0] == "method":
                self.set_method(param[3])
            elif param[0] == "keepClusterMapping":
                self.set_keep_cluster_mapping(param[3])

    def d2_plot(self, last_sequenz: List[int], costumes: List[Costume]) -> None:
        pass
