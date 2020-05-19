from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import numpy as np
from sklearn import manifold
from backend.logger import Logger, LogLevel
from backend.entity import Costume
from typing import List
from matplotlib import pyplot as plt
import re
from matplotlib.collections import LineCollection

class ScalingType(enum.Enum):
    mds = 0     # metric multidimensional scaling

    """
    Returns the name of the given ScalingType.
    """
    @staticmethod
    def get_name(scalingType) -> str:
        name = ""
        if scalingType == ScalingType.mds:
            name = "mds"
        else:
            Logger.error("No name for scaling \"" + str(scalingType) + "\" specified")
            raise ValueError("No name for scaling \"" + str(scalingType) + "\" specified")
        return name

    """
    Returns the description of the given ScalingType.
    """
    @staticmethod
    def get_description(scalingType) -> str:
        description = ""
        if scalingType == ScalingType.mds:
            description = ("Multidimensioal scaling (MDS) is a means of visualizing"
                    + "the level of similarity of individual cases of dataset." 
                    + "MDS is used to translate 'information about the pairwise" 
                    + "distances among a set of n objects or individuals" 
                    + "into a configuration of n points mapped into an abstract Cartesian space.")
        else:
            Logger.error("No description for scaling \"" + str(scalingType) + "\" specified")
            raise ValueError("No description for scaling \"" + str(scalingType) + "\" specified")
        return description

class Scaling(metaclass=ABCMeta):
    """
    Interface for Scaling Object
    """
    @abstractmethod
    def scaling(self, similarity_matrix: np.matrix ) -> np.matrix:
        pass
    @abstractmethod
    def stress_level(self) -> int:
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
class ScalingFactory:
    
    @staticmethod
    def create(type: ScalingType) -> Scaling:
        if type == ScalingType.mds:
            return MultidimensionalScaling()
        else:
            Logger.error("Unknown type of scaling. The application will quit know.")
            raise Exception("Unknown type of scaling.")

"""
multidimensional scaling
"""
class MultidimensionalScaling(Scaling):
    """
    Multidimensional scaling Referenz: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    Parameters:
        n_components/dimensions:int, optional, default: 2
                                Number of dimensions in which to immerse the dissimilarities.

        metric:                 boolean, optional, default: True
                                If True, perform metric MDS; otherwise, perform nonmetric MDS.

        n_init/repeatSMACOF:    int, optional, default: 4
                                Number of times the SMACOF algorithm will be run with different initializations. 
                                The final results will be the best output of the runs, determined by the run with
                                the smallest final stress.

        max_iter:               int, optional, default: 300
                                Maximum number of iterations of the SMACOF algorithm for a single run.

        eps:                    float, optional, default: 1e-3
                                Relative tolerance with respect to stress at which to declare convergence.

        n_jobs:                 int or None, optional (default=None)
                                The number of jobs to use for the computation. If multiple initializations are used
                                (n_init), each run of the algorithm is computed in parallel.
                                None means 1 unless in a joblib.parallel_backend context.
                                -1 means using all processors. See Glossary for more details.

        random_state:           int, RandomState instance or None, optional, default: None
                                The generator used to initialize the centers. If int, random_state is the seed used 
                                by the random number generator; If RandomState instance, random_state is the random 
                                number generator; If None, the random number generator is the RandomState instance 
                                used by np.random.

        dissimilarity:          ‘euclidean’ | ‘precomputed’, optional, default: ‘euclidean’
                                Dissimilarity measure to use:

                                ‘euclidean’:
                                Pairwise Euclidean distances between points in the dataset.

                                ‘precomputed’:
                                Pre-computed dissimilarities are passed directly to fit and fit_transform.
    """
    def __init__(
        self,
        dimensions: int = 2,
        repeatSMACOF: int = 4,
        max_iter: int = 300,
        eps: float = 1e-3,
        random_state: np.random.RandomState =  np.random.RandomState(seed=3),
        dissimilarity: str = "euclidean",
        n_jobs: int = 1
    ) -> None :
        self.__stress: float = 0
        self.__bool_create_mds: bool = False
        self.__similarity_matrix: np.matrix
        self.__position_matrix: np.matrix
        self.__dimensions: int = dimensions
        self.__repeatSMACOF: int = repeatSMACOF
        self.__max_iter: int = max_iter
        self.__eps: float = eps
        self.__random_state: np.random.RandomState = random_state
        self.__dissimilarity: str = dissimilarity
        self.__n_jobs: int = 1
        try:
            self.__mds: manifold._mds.MDS = manifold.MDS(   n_components    = self.__dimensions,
                                                            n_init          = self.__repeatSMACOF,
                                                            max_iter        = self.__max_iter,
                                                            eps             = self.__eps,
                                                            random_state    = self.__random_state,
                                                            dissimilarity   = self.__dissimilarity,
                                                            n_jobs          = self.__n_jobs)
        except Exception as error:
            Logger.error("An Exception occurs by MDS initialization: " + str(error))
            raise Exception("Exception occurs by MDS initialization.")
        
    # getter methodes
    def get_dimensions(self) ->  int:
        return self.__dimensions
    def get_repeatSMACOF(self) -> int:
        return self.__repeatSMACOF
    def get_max_iter(self) -> int:
        return self.__max_iter
    def get_dissimilarity(self) -> str:
        return self.__dissimilarity
    def get_n_jobs(self) -> int:
        return self.__n_jobs
    def get_eps(self) -> float:
        return self.__eps

    # setter methodes
    def set_dimensions(self, dimensions: int = 2 ) -> None:
        self.__dimensions = dimensions
    def set_repeatSMACOF(self, repeatSMACOF: int = 4) -> None:
        self.__repeatSMACOF = repeatSMACOF
    def set_max_iter(self, max_iter: int = 3000) -> None:
        self.__max_iter = max_iter
    def set_dissimilarity(self, dissimilarity: str = "precomputed") -> None:
        self.__dissimilarity = dissimilarity
    def set_n_jobs(self, n_jobs: int = 1) -> None:
        self.__n_job = n_jobs
    def set_eps(self, eps: float = 1e-3) -> None:
        self.__eps = eps
  
    # methodes from interface
    def scaling(self, similarity_matrix: np.matrix ) -> np.matrix:
        try:
            self.__mds = manifold.MDS(  n_components    = self.__dimensions,
                                        n_init          = self.__repeatSMACOF,
                                        max_iter        = self.__max_iter,
                                        eps             = self.__eps,
                                        random_state    = self.__random_state,
                                        dissimilarity   = self.__dissimilarity,
                                        n_jobs          = self.__n_jobs)
            self.__mds = self.__mds.fit(similarity_matrix)
            self.__position_matrix = self.__mds.embedding_
            self.__stress = self.__mds.stress_
            self.__bool_create_mds = True
            self.__similarity_matrix = similarity_matrix
            return self.__position_matrix
        except Exception as error:
            Logger.error("An Exception occurs by MDS initialization: " + str(error))
            raise Exception("Exception occurs in Method scaling by MDS initialization.")
  
    def stress_level(self) -> int:
        if self.__bool_create_mds:
            return round(self.__stress,3)
        else:
            Logger.error("No similarity matrix was scaled.")
            raise Exception("No similarity matrix was scaled")

    def d2_plot(self,last_sequenz: List[int] , costumes: List[Costume]) -> None:
        _Plots_in_scaling._two_dim_plot(self.__similarity_matrix,self.__position_matrix,last_sequenz,costumes)
        
# setter and getter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        ScalingTypeName = "Multidimensional Scaling"
        params.append(("name", "ScalingTyp" ,"description", ScalingTypeName ,"header"))
        parameter_dimensions = self.get_dimensions()
        params.append(("dimensions", "Dimensions" ,"description", parameter_dimensions, "number", 1 , 1 ))
        parameter_repeatSMACOF = self.get_repeatSMACOF()
        params.append(("repeatSMACOF", "repeat SMACOF Algorithm" ,"description", parameter_repeatSMACOF, "number", 1,1 ))
        parameter_maxIter = self.get_max_iter()
        params.append(("maxIter", "Number Iterations" ,"description", parameter_maxIter , "number" , 1,1))
        parameter_dissimilarity = self.get_dissimilarity()
        params.append(("dissimilarity" , "Dissimilarity" ,"description", parameter_dissimilarity, "select",("precomputed", "euclidean")))
        parameter_eps = self.get_eps()
        params.append(("eps","Epsilon","description", parameter_eps, "number" ,0,0.000000001 ))
        return params

    def set_param_list(self, params: list = []) -> np.matrix:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        for param in params:
            if param[0] == "dimensions":
                self.set_dimensions(param[3])
            elif param[0] == "repeatSMACOF":
                self.set_repeatSMACOF(param[3])
            elif param[0] == "maxIter":
                self.set_max_iter(param[3])
            elif param[0] == "dissimilarity":
                self.set_dissimilarity(param[3])
            elif param[0] == "eps":
                self.set_eps(param[3])
            

"""
class manage the plots
"""
class _Plots_in_scaling():

    @staticmethod
    def _two_dim_plot(similarity_matrix: np.matrix, position_matrix: np.matrix, last_sequenz: List[int], costumes: List[Costume] ) -> None:
        if len(position_matrix[0]) != 2:
            Logger.error("Dimension of Position Matrix is not 2!")
            raise Exception("Dimension of Position Matrix is not 2!")

        plt.figure(3)
        ax = plt.axes([0., 0., 1., 1.])
        s = 100
        plt.scatter(position_matrix[:, 0], position_matrix[:, 1], color='turquoise', s=s, lw=0, label='MDS')
        plt.legend(scatterpoints=1, loc='best', shadow=False)
        EPSILON = np.finfo(np.float32).eps
        #print("similarities.max")
        #print(similarities.max())
        similarity_matrix = similarity_matrix.max() / (similarity_matrix + EPSILON) * 100
        #print("similarities after max/(sim+eps)*100")
        #print(similarities)
        np.fill_diagonal(similarity_matrix, 0)
        # Plot the edges
        #start_idx, end_idx = np.where(pos)
        # a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[position_matrix[i, :], position_matrix[j, :]]
                    for i in range(len(position_matrix)) for j in range(len(position_matrix))]
        #print("segments")
        #print(segments)
        values = np.abs(similarity_matrix)
        #print("Values")
        #print(values)
        lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.Blues,
                        norm=plt.Normalize(0, values.max()))
        lc.set_array(similarity_matrix.flatten())
        lc.set_linewidths(np.full(len(segments), 0.5))
        ax.add_collection(lc)
        # describe points
        style = dict(size=7, color='black')
        count: int = 0
        for i in last_sequenz:
            txt = str(i)+". " +str(costumes[i])
            txt = re.sub("(.{20})", "\\1-\n", str(txt), 0, re.DOTALL)
            plt.annotate(txt, (position_matrix[count, 0], position_matrix[count, 1]), **style)
            count += 1
        #plt.ylim((-0.6,0.6))
        #plt.xlim((-0.5,0.5))
        plt.draw()