import enum
from qhana.backend.logger import Logger
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
from qhana.backend.entity import Costume
import math
from qhana.backend.attribute import Attribute


"""
Enum for Labelers
"""


class LabelerTypes(enum.Enum):
    fixedSubset = 0  # labeler based on subsets' positive/negative classes
    attribute = 1 # labeler that uses an attribute to label data
    clusters = 2 # labeler based on the clusters from clustering

    @staticmethod
    def get_name(labelerType) -> str:
        name = ""
        if labelerType == LabelerTypes.fixedSubset:
            name = "fixedSubset"
        elif labelerType == LabelerTypes.attribute:
            name = "attribute"
        elif labelerType == LabelerTypes.clusters:
            name = "clusters"
        else:
            Logger.error("No name for labeler \"" + str(labelerType))
            raise ValueError("No name for labeler \"" + str(labelerType))
        return name

    """
    Returns the description of the given LabelerType.
    """

    @staticmethod
    def get_description(labelerType) -> str:
        description = ""
        if labelerType == LabelerTypes.fixedSubset:
            description = ("labeler based on subsets' positive/negative classes")
        elif labelerType == LabelerTypes.attribute:
            description = ("labeler based on an attribute")
        elif labelerType == LabelerTypes.clusters:
            description = ("Labeler based on clusters determined in clustering")
        else:
            Logger.error("No description for labeler \"" + str(labelerType) + "\" specified")
            raise ValueError("No description for labeler \"" + str(labelerType) + "\" specified")
        return description


class Labeler(metaclass=ABCMeta):
    """
    Interface for Labeler Object
    """

    @abstractmethod
    def get_labels(self, position_matrix : np.matrix, entities: List, similarity_matrix : np.matrix) -> (list, dict):
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
Represents the factory to create a labeler object
"""

class LabelerFactory:

    @staticmethod
    def create(type: LabelerTypes) -> Labeler:
        if type == LabelerTypes.fixedSubset:
            return fixedSubsetLabeler()
        elif type == LabelerTypes.attribute:
            return attributeLabeler()
        elif type == LabelerTypes.clusters:
            return clustersLabeler()
        else:
            Logger.error("Unknown type of labeler. The application will quit know.")
            raise Exception("Unknown type of labeler.")


class fixedSubsetLabeler(Labeler):

    def __init__(
        self
    ):
        return

    def get_labels(self, position_matrix : np.matrix, entities: List, similarity_matrix : np.matrix) -> (list, dict):
        validIds = [entity.id for entity in entities]

        # TODO: selected subset size could be passed as a parameter
        subsetSizes = np.array([5, 10, 25, 40])
        maxId = max(validIds)
        larger_subsets = subsetSizes > maxId
        if not True in larger_subsets:
            raise Exception("Error in labeler: no valid subset")
        n_samples = subsetSizes[list(larger_subsets).index(True)] # gets the first subsetSize that is larger than maxId

        rawLabels = [1 for _ in range(n_samples)]
        for i in range(math.ceil(n_samples / 2), n_samples):
            rawLabels[i] = 0

        validLabels = np.array(rawLabels)[validIds]

        labels = validLabels
        dict_label_class = {1: 'positive', 0: 'negative'}
        return labels, dict_label_class

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "Fixed subset labeler"
        params.append(("name", "Labeler Type" , "Generates the labels for pre-defined subsets based on the order of data points "\
                                                +"The first half of data points is associated with the positive class, the second half "\
                                                +"with the negative class. If number of data points is uneven, then the positive class has "\
                                                +"one more data point than negative class.", labelerTypeName , "header"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

class attributeLabeler(Labeler):

    def __init__(
        self,
        attribute = Attribute.geschlecht
    ):
        self.__attribute = attribute
        return

    def get_labels(self, position_matrix : np.matrix, entities: List, similarity_matrix : np.matrix) -> (list, dict):
        rawLabels = [entity.values[self.__attribute][0] for entity in entities]
        classesSet = set(rawLabels)
        classes = list(classesSet)
        n_classes = len(classes)

        # transform to numeric labels
        labels = [classes.index(rawLabel) for rawLabel in rawLabels]

        # create a dictionary number -> label
        dict_label_class = {}
        for i in range(n_classes):
            dict_label_class[i] = classes[i]
            
        labels = np.array(labels)

        return labels, dict_label_class

    def get_attribute(self):
        return self.__attribute

    def set_attribute(self, attribute):
        self.__attribute = attribute

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "labeler by attribute"
        params.append(("name", "Labeler Type" , "Generates the labels based on an attribute ",
                                labelerTypeName , "header"))

        parameter_attribute = self.get_attribute().value
        description_attribute = "Attribute : The attribute to be used for labeling (default=gender) "
        params.append(("attribute", "Attribute", description_attribute, parameter_attribute, "select", [attribute.value for attribute in Attribute]))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "attribute":
                for attr in Attribute:
                    if attr.value == param[3]:
                        self.set_attribute(attr)
                        break

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

class clustersLabeler(Labeler):

    def __init__(
        self,
        attribute = Attribute.geschlecht
    ):
        self.__attribute = attribute
        return

    def get_labels(self, position_matrix : np.matrix, entities: List, similarity_matrix : np.matrix) -> (list, dict):
        raise Exception("Wrong usage of clusters labeler type: Labels need to be passed from clustering step instead of calling the get_labels method")

    def get_attribute(self):
        return self.__attribute

    def set_attribute(self, attribute):
        self.__attribute = attribute

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        labelerTypeName = "labeler by clusters"
        params.append(("name", "Labeler Type" , "Generates the labels based on clusters determined during clustering ",
                                labelerTypeName , "header"))
        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

