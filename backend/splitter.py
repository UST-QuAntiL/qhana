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
from backend.classification import get_dict_dataset

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
            name = "proportionalRandom"
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
    def get_train_test_set(self, position_matrix : np.matrix , labels: list, entities: List, similarity_matrix : np.matrix) \
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

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, entities: List, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
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
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, entities: List, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        n_samples = position_matrix.shape[0]
        n_classes = len(set(labels))

        dict_class_data = get_dict_dataset(position_matrix, labels)
        # determine how many training samples to get from each class, this should be proportional to the overall selected data
        n_train_per_class = [math.ceil(len(dict_class_data[_class])*self.train_set_size/n_samples) for _class in set(labels)]

        if n_samples <= self.train_set_size:
            Logger.error("Train set size in splitter is larger than number of samples.")
            raise Exception("Train set size in splitter is larger than number of samples.")
        elif n_samples <= sum(n_train_per_class):
            Logger.error("Failed to compute a reasonable splitting with train set proportinal to the selected data set.")
            raise Exception("Failed to compute a reasonable splitting with train set proportinal to the selected data set.")

        current_hash = hash(str(position_matrix)+str(self.train_set_size))
        if not (self.reuse and current_hash == self.matrix_hash):
            self.matrix_hash = current_hash
            self.train_set = []
            self.test_set = []
            self.train_labels = []
            self.test_labels = []

            for i in range(n_classes):
                lbl = (list(set(labels)))[i]
                # get n_train_per_class random items from current class and add to train set
                class_data = np.array(dict_class_data[lbl])
                
                i_train_indices = random.sample(range(len(class_data)), n_train_per_class[i])
                i_train = class_data[i_train_indices]
                i_test_indices = setdiff1d(range(len(class_data)), i_train_indices)
                i_test = class_data[i_test_indices]

                self.train_set.extend(i_train)
                self.test_set.extend(i_test)
                self.train_labels.extend([lbl for _ in range(len(i_train_indices))])
                self.test_labels.extend([lbl for _ in range(len(i_test_indices))])

        return np.array(self.train_set), np.array(self.train_labels),\
                np.array(self.test_set), np.array(self.test_labels)

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
                                                +"as train data and the rest as test data. "\
                                                +"Aims at choosing train data approx. proportionally to the data set.",
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

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, entities: List, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
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
        labelerTypeName = "sklearn splitter"
        params.append(("name", "Splitter Type" , "Directly uses the train_test_split method of sklearn to split data into train and test sets ",
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
        train_subset = 10
    ):
        self.__train_subset = train_subset
        self.__subsetSizes = np.array([5, 10, 25, 40])

    def get_train_test_set(self, position_matrix : np.matrix, labels: list, entities: List, similarity_matrix : np.matrix) -> (np.matrix, list, np.matrix, list):
        validIds = [entity.id for entity in entities]

        # TODO: selected subset size could be passed as a parameter
        subsetSizes = np.array([5, 10, 25, 40])
        maxId = max(validIds)
        larger_subsets = subsetSizes > maxId
        if not True in larger_subsets:
            raise Exception("Error in labeler: no valid subset")
        n_samples = subsetSizes[list(larger_subsets).index(True)] # gets the first subsetSize that is larger than maxId

        if n_samples <= self.__train_subset:
            raise Exception("Error in splitter: train subset not smaller than initialized subset")
        else:
            train_indices = []
            for _class in {'Negative', 'Positive'}:
                fileTrainSet = open('subsets/{}/{}Subset{}.csv'.format(str(self.__train_subset),_class, str(self.__train_subset)), 'r')
                LinesTrainSet = fileTrainSet.readlines()[1:]
                fileInitSet = open('subsets/{}/{}Subset{}.csv'.format(str(n_samples),_class,str(n_samples)), 'r')
                LinesInitSet = fileInitSet.readlines()[1:]
                for line in LinesTrainSet:
                    train_indices.append(LinesInitSet.index(line) \
                        + (0 if _class == 'Positive' else math.ceil(n_samples/2)))

        test_indices = setdiff1d(range(n_samples), train_indices, assume_unique=True)

        # translate indices to the corresponding ones in position matrix
        # meanwhile invalid samples are ignored
        train_indices_new, test_indices_new = [], []
        for entity in entities:
            if entity.id in train_indices:
                train_indices_new.append(list(entities).index(entity))
            if entity.id in test_indices:
                test_indices_new.append(list(entities).index(entity))

        return position_matrix[train_indices_new], labels[train_indices_new], \
                position_matrix[test_indices_new], labels[test_indices_new]

    def get_trainsubset(self):
        return self.__train_subset

    def set_trainsubset(self, train_subset):
        self.__train_subset = train_subset

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "Subset splitter"
        params.append(("name", "Splitter Type" , "Use a smaller subset as train set and the remaining data as test set ",
                                                labelerTypeName , "header"))

        parameter_trainsubset = self.get_trainsubset()
        description_trainsubset = "Train subset : The subset that shall be used for training. Must be smaller than the initialized subset. "
        params.append(("trainsubset", "Train subset", description_trainsubset, parameter_trainsubset, "select", [value for value in self.__subsetSizes]))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "trainsubset":
                self.set_trainsubset(int(param[3]))

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass
