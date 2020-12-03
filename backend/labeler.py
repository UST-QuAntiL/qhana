import enum
from backend.logger import Logger
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
from backend.entity import Costume
import math
from backend.attribute import Attribute


"""
Enum for Labelers
"""


class LabelerTypes(enum.Enum):
    fixedSubset = 0  # labeler based on subsets' positive/negative classes
    attribute = 1 # labeler that uses an attribute to label data

    @staticmethod
    def get_name(labelerType) -> str:
        name = ""
        if labelerType == LabelerTypes.fixedSubset:
            name = "fixedSubset"
        elif labelerType == LabelerTypes.attribute:
            name = "attribute"
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
        else:
            Logger.error("No description for labeler \"" + str(labelerType) + "\" specified")
            raise ValueError("No description for labeler \"" + str(labelerType) + "\" specified")
        return description


class Labeler(metaclass=ABCMeta):
    """
    Interface for Labeler Object
    """

    @abstractmethod
    def get_labels(self, position_matrix : np.matrix, entities: List, attributes: dict, similarity_matrix : np.matrix) -> (list, dict):
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
        else:
            Logger.error("Unknown type of labeler. The application will quit know.")
            raise Exception("Unknown type of labeler.")


class fixedSubsetLabeler(Labeler):

    def __init__(
        self
    ):
        return

    def get_labels(self, position_matrix : np.matrix, entities: List, attributes: dict, similarity_matrix : np.matrix) -> (list, dict):
        n_samples = len(position_matrix)
        labels = [1 for _ in range(n_samples)]
        for i in range(math.ceil(n_samples / 2), n_samples):
            labels[i] = -1

        labels = np.array(labels)
        dict_label_class = {1: 'positive', -1: 'negative'}
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
                                                +"The first half of data points is associated with the +1 class, the second half "\
                                                +"with the -1 class. If number of data points is uneven, then the +1 class has "\
                                                +"one more data point than -1 class.", labelerTypeName , "header"))

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

    def get_labels(self, position_matrix : np.matrix, entities: List, attributes: dict, similarity_matrix : np.matrix) -> (list, dict):
        rawLabels = [entity.values[self.__attribute][0] for entity in entities]
        classesSet = set(rawLabels)
        classes = list(classesSet)
        n_classes = len(classes)

        labels = [classes.index(rawLabel) for rawLabel in rawLabels]

        dict_label_class = {}
        for i in range(n_classes):
            dict_label_class[i] = classes[i]
            
        """ if there are only 2 classes: use -1 and 1 as labels"""
        if n_classes == 2:
            labels = np.array(labels)
            labels = np.where(labels==0, -1, labels)
            dict_label_class[-1] = dict_label_class[0]

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


