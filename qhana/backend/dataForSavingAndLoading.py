from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import numpy as np
from typing import List
import backend.scaling as scal
import backend.clustering as clu
#import backend.similarities as sim
import backend.entitySimilarities as esim

class DataForSavingAndLoading(metaclass=ABCMeta):
    """
    Interface for DataForSaving Object
    """

class DataForCostumePlan(DataForSavingAndLoading):
    def __init__(
        self,
        costumeplan_object: List = None,
    ) -> None:
        self.__costumeplan_object: List = costumeplan_object

    # getter methodes
    def get_object(self) -> List:
        return self.__costumeplan_object

class DataForEntitySimilarities(DataForSavingAndLoading):
    def __init__(
        self,
        similarities_object: esim.EntitySimilarities = None,
    ) -> None:
        self.__similarities_object: esim.EntitySimilarities = similarities_object

    # getter methodes
    def get_object(self) -> esim.EntitySimilarities:
        return self.__similarities_object
"""
class DataForCostumeSimilarities(DataForSavingAndLoading):
    def __init__(
        self,
        similarities_object: sim.Similarities = None,
    ) -> None:
        self.__similarities_object = similarities_object

    # getter methodes
    def get_object(self) -> sim.Similarities:
        return self.__similarities_object
"""
class DataForScaling(DataForSavingAndLoading):
    def __init__(
        self,
        scaling_object: scal.Scaling = None,
    ) -> None:
        self.__scaling_object = scaling_object

    # getter methodes
    def get_object(self) -> scal.Scaling:
        return self.__scaling_object

class DataForClustering(DataForSavingAndLoading):
    def __init__(
        self,
        cluster_object: clu.Clustering = None,
    ) -> None:
        self.__cluster_object = cluster_object

    # getter methodes
    def get_object(self) -> clu.Clustering:
        return self.__cluster_object