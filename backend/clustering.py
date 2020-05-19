from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import numpy as np
from backend.logger import Logger, LogLevel
from sklearn.cluster import OPTICS
from backend.entity import Costume
from typing import List


"""
Enums for Clustertyps
"""
class ClusteringType(enum.Enum):
    optics = 0  # OPTICS =:Ordering Points To Identify the Clustering Structure 

    @staticmethod
    def get_name(clusteringTyp) -> str:
        name = ""
        if clusteringTyp == ClusteringType.optics :
            name = "optics"
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
        else:
            Logger.error("No description for clustering \"" + str(clusteringTyp) + "\" specified")
            raise ValueError("No description for clustering \"" + str(clusteringTyp) + "\" specified")
        return description


class Clustering(metaclass=ABCMeta):
    """
    Interface for Scaling Object
    """
    @abstractmethod
    def create_cluster(self, position_matrix : np.matrix ) -> np.matrix:
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

""" 
Represents the factory to create an scaling object
"""
class ClusteringFactory:
    
    @staticmethod
    def create(type: ClusteringType) -> Clustering:
        if type == ClusteringType.optics:
            return Optics()
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

    def create_cluster(self, position_matrix : np.matrix ) -> np.matrix:
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
        params.append(("name", "ClusterTyp" ,"description", clusteringTypeName ,"header"))
        parameter_minSamples = self.get_min_samples()
        params.append(("minSamples", "min Samples" ,"description", parameter_minSamples, "number", 1 , 1 ))
        parameter_maxEps = self.get_max_eps()
        params.append(("maxEps", "max Epsilon" ,"description", parameter_maxEps, "text" ))
        parameter_metric = self.get_metric()
        params.append(("metric", " Metric" ,"description", parameter_metric , "select" , ('precomputed','minkowski','cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan')))
        parameter_p = self.get_p()
        params.append(("p" , "Parameter p for minkowski" ,"description", parameter_p, "number" , 1,1 ))
        parameter_cluster_method = self.get_cluster_method()
        params.append(("cluster_method","Cluster Method","description", parameter_cluster_method, "select" , ("xi" , "dbscan")))
        parameter_eps = self.get_eps()
        params.append(("eps", "Epsilon","description", parameter_eps , "text"))
        parameter_xi = self.get_xi()
        params.append(("xi","Xi" ,"description", parameter_xi, "number" , 0, 0.001))
        parameter_predecessor_correction = self.get_predecessor_correction()
        params.append(("predecessor_correction", "Predecessor Correction" ,"description", parameter_predecessor_correction, "checkbox"))
        parameter_min_cluster_size = self.get_min_cluster_size()
        params.append(("min_cluster_size","Min Cluster Size","description", parameter_min_cluster_size, "text"))
        parameter_algorithm = self.get_algorithm()
        params.append(("algorithm","Algorithm" ,"description", parameter_algorithm , "select" , ('auto', 'ball_tree', 'kd_tree', 'brute')))
        parameter_leaf_size = self.get_leaf_size()
        params.append(("leaf_size","Leaf Size" ,"description", parameter_leaf_size, "number", 0 , 1))
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


                
            
            



