import enum
from backend.logger import Logger
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
from backend.entity import Costume
import math
import random
from numpy import setdiff1d
from sklearn.model_selection._split import train_test_split
from random import randrange

"""
Enum for Splitters
"""


class SplitterTypes(enum.Enum):
    none = 0 # train set = test set = all data
    random = 1  # splitter that selects train set randomly
    sklearn = 2 # sklearn splitter
    subset = 3 # splitter that selects a smaller subset (of the pre-defined ones) as train set

    @staticmethod
    def get_name(splitterType) -> str:
        name = ""
        if splitterType == SplitterTypes.none:
            name = "none"
        elif splitterType == SplitterTypes.random:
            name = "random"
        elif splitterType == SplitterTypes.sklearn:
            name = "sklearn"
        elif splitterType == SplitterTypes.subset:
            name = "subset"
        else:
            Logger.error("No name for labeler \"" + str(splitterType))
            raise ValueError("No name for labeler \"" + str(splitterType))
        return name

    """
    Returns the description of the given SplitterType.
    """

    @staticmethod
    def get_description(splitterType) -> str:
        description = ""
        if splitterType == SplitterTypes.none:
            description = ("train set = test set = all data")
        elif splitterType == SplitterTypes.random:
            description = ("splitter that selects train set randomly")
        elif splitterType == SplitterTypes.sklearn:
            description = ("sklearn splitter")
        elif splitterType == SplitterTypes.subset:
            description = ("splitter that selects a smaller subset (of the pre-defined ones) as train set")
        else:
            Logger.error("No description for splitter \"" + str(splitterType) + "\" specified")
            raise ValueError("No description for splitter \"" + str(splitterType) + "\" specified")
        return description


class Splitter(metaclass=ABCMeta):
    """
    Interface for Splitter Object
    """

    @abstractmethod
    def get_train_test_set(self, position_matrix : np.matrix , labels: list, similarity_matrix : np.matrix) \
                                -> (np.matrix, list, np.matrix, list):
        pass

    @abstractmethod
    def get_param_list(self) -> list:
        pass

    @abstractmethod
    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    @abstractmethod
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

"""
Represents the factory to create a splitter object
"""

class SplitterFactory:

    @staticmethod
    def create(type: SplitterTypes) -> Splitter:
        if type == SplitterTypes.none:
            return noneSplitter()
        elif type == SplitterTypes.random:
            return randomSplitter()
        elif type == SplitterTypes.sklearn:
            return sklearnSplitter()
        elif type == SplitterTypes.subset:
            return subsetSplitter()
        else:
            Logger.error("Unknown type of splitter. The application will quit know.")
            raise Exception("Unknown type of splitter.")


class noneSplitter(Splitter):

    def __init__(
        self,
    ):
        return

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        return position_matrix, labels, position_matrix, labels

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "None splitter"
        params.append(("name", "Splitter Type" , "Use all data for both training and testing ",
                                                labelerTypeName , "header"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

class randomSplitter(Splitter):

    def __init__(
        self,
        reuse = True,
        train_set_size = 10
    ):
        self.reuse = reuse
        self.train_set_size = train_set_size
        return

    matrix_hash = None
    train_indices = []

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        n_samples = position_matrix.shape[0]

        print(position_matrix.shape)
        print(n_samples)

        if n_samples <= self.train_set_size:
            self.train_indices = []
            Logger.error("Train set size in splitter is larger than number of samples.")
            raise Exception("Train set size in splitter is larger than number of samples.")

        current_hash = hash(str(position_matrix)+str(self.train_set_size))
        print(current_hash)
        print(self.matrix_hash)
        if not (self.reuse and current_hash == self.matrix_hash):
            self.matrix_hash = current_hash
    
            # select positive samples' indices
            train_set = []
            while len(set(train_set)) < math.ceil(self.train_set_size/2):
                train_set.append(random.randrange(0, math.ceil(n_samples / 2)))
                
            # add negative samples' indices
            while len(set(train_set)) < self.train_set_size:
                train_set.append(random.randrange(math.ceil(n_samples / 2), n_samples))
            self.train_indices = sorted(list(set(train_set)))
            print(self.train_indices)

        test_indices = setdiff1d(range(n_samples), self.train_indices, assume_unique=True)

        return position_matrix[self.train_indices], labels[self.train_indices], position_matrix[test_indices], labels[test_indices]

    def get_train_set_size(self):
        return self.train_set_size

    def set_train_set_size(self, train_set_size):
        self.train_set_size = train_set_size

    def get_reuse(self):
        return self.reuse

    def set_reuse(self, reuse):
        self.reuse = reuse

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "Random splitter"
        params.append(("name", "Splitter Type" , "Custom made random splitter: Selects a number of random items from the data set "\
                                                +"as train data and the rest as test data "\
                                                +"half of the train data will be in the +1 class, the rest in the -1 class.", 
                                                labelerTypeName , "header"))

        parameter_trainsetsize = self.get_train_set_size()
        description_trainsetsize = "Train set size : int (default=10)\n Size of the train set"
        params.append(("trainsetsize", "Train set size", description_trainsetsize, parameter_trainsetsize, "number", 1, 1))
        
        parameter_reuse = self.get_reuse()
        description_reuse = "reuse: bool, (default=True)\n If True: the previous split is used again for the same data set"
        params.append(("reuse", "Reuse" , description_reuse, parameter_reuse, "checkbox"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "trainsetsize":
                self.set_train_set_size(param[3])
            if param[0] == "reuse":
                self.set_reuse(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

class sklearnSplitter(Splitter):

    def __init__(
        self,
        train_size = 0.25,
        reuse = True
    ):
        self.train_size = train_size
        self.reuse = reuse
        self.rand_state = randrange(100)
        return

    rand_state = None

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        if not self.reuse:
            self.rand_state = randrange(100) 

        train_X, test_X, train_y, test_y = train_test_split(position_matrix, labels, train_size=self.train_size, random_state=self.rand_state)

        return train_X, train_y, test_X, test_y

    def get_train_size(self):
        return self.train_size

    def set_train_size(self, train_size):
        self.train_size = train_size

    def get_reuse(self):
        return self.reuse

    def set_reuse(self, reuse):
        self.reuse = reuse

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "None splitter"
        params.append(("name", "Splitter Type" , "Use all data for both training and testing ",
                                                labelerTypeName , "header"))
        
        parameter_trainsize = self.get_train_size()
        description_trainsize = "trainsize: float or int, (default=0.25)\n If float: represents the proportion of the dataset to include in the train split"\
                                +"If int, represents the absolute number of train samples"
        params.append(("trainsize", "Train size" , description_trainsize, parameter_trainsize, "number", 0, 0.01))

        parameter_reuse = self.get_reuse()
        description_reuse = "reuse: bool, (default=True)\n If True: the previous split is used again for the same data set"
        params.append(("reuse", "Reuse" , description_reuse, parameter_reuse, "checkbox"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "trainsize":
                if param[3] > 1:
                    self.set_train_size(int(param[3]))
                else:
                    self.set_train_size(param[3])
            if param[0] == "reuse":
                self.set_reuse(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

class subsetSplitter(Splitter):

    def __init__(
        self,
    ):
        return

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        return position_matrix, labels, position_matrix, labels

        # TODO: Identify smaller subset and return it as train set
        len = position_matrix.shape[0]
        if len <= 10:
            return position_matrix
        else:
            for _class in {'Negative', 'Positive'}:
                file10 = open('subsets/10/{}Subset10.csv'.format(_class), 'r')
                Lines10 = file10.readlines()[1:]
                fileSet = open('subsets/{}/{}Subset{}.csv'.format(str(len),_class,str(len)), 'r')
                LinesSet = fileSet.readlines()[1:]
                for line in Lines10:
                    print(LinesSet.index(line))

        return

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "None splitter"
        params.append(("name", "Splitter Type" , "Use all data for both training and testing ",
                                                labelerTypeName , "header"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass
