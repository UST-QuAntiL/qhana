from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import networkx as nx
from networkx import Graph
from qhana.backend.taxonomie import Taxonomie
from qhana.backend.logger import Logger
import numpy as np
from qhana.backend.logger import Logger
import os
import json
import math
from qhana.backend.timer import Timer

""" 
Defines an enum to list up all available element comparer
"""
class ElementComparerType(enum.Enum):
    wuPalmer = "wuPalmer"
    timeTanh = "timeTanh"

    """
    Returns the name of the given ElementComparerType.
    """
    @staticmethod
    def get_name(elementComparerType) -> str:
        name = ""
        if elementComparerType == ElementComparerType.wuPalmer:
            name += "WuPalmer"
        elif elementComparerType == ElementComparerType.timeTanh:
            name += "TimeTanh"
        else:
            Logger.error("No name for element comparer \"" + str(elementComparerType) + "\" specified")
            raise ValueError("No name for element comparer \"" + str(elementComparerType) + "\" specified")
        return name

    """
    Returns the description of the given ElementComparerType.
    """
    @staticmethod
    def get_description(elementComparerType) -> str:
        description = ""
        if elementComparerType == ElementComparerType.wuPalmer:
            description += "Compares two elements based on a taxonomy " \
                + "using the wu palmer similarity measure."
        elif elementComparerType == ElementComparerType.timeTanh:
            description += "Compares two timecodes using the tanh function: " \
                + "tanh(abs(a-b) / 7200). We normalize this function to 7200 seconds."
        else:
            Logger.error("No description for element comparer \"" + str(elementComparerType) + "\" specified")
            raise ValueError("No description for element comparer \"" + str(elementComparerType) + "\" specified")
        return description

""" 
Represents the abstract element comprarer base class
"""
class ElementComparer(metaclass=ABCMeta):
    """ 
    Returns the comparison value of first and second
    element based on the giben base
    """
    @abstractmethod
    def compare(self, first: Any, second: Any, base: Any) -> float:
        pass

    """
    Creates a full cache on file, i.e. calculates all pariwise similarities
    and safes the result in a file
    """
    @abstractmethod
    def create_cache(self, base: Any) -> None:
        pass
""" 
Represents the factory to create an element comparer
"""
class ElementComparerFactory:
    """
    Static method for creating an element comparer
    """
    @staticmethod
    def create(type: ElementComparerType) -> ElementComparer:
        if type == ElementComparerType.wuPalmer:
            return WuPalmer()
        elif type == ElementComparerType.timeTanh:
            return TimeTanh()
        else:
            raise Exception("Unknown type of element comparer")
        return

"""
Represents the conventional wu palmer similarity measure
"""
class WuPalmer(ElementComparer):

    """
    Constructor.
    """
    def __init__(self):
        self.cache = None
        return

    """
    Applies try to use cache first if available, if not,
    run compare_inner
    """
    def compare(self, first: str, second: str, base: Taxonomie) -> float:
        # check if cache is available
        cache = dict()
        if self.cache is None:
            if os.path.isdir("cache") != False:
                fileName = "cache/" + base.name + ".json"
                if os.path.isfile(fileName) != False:
                    self.cache = self.__loads_json(fileName)
                    cache = self.cache
        else:
            cache = self.cache

        if (first, second) in cache:
            if (first, second) in cache:
                return self.cache[(first, second)]

        return self.compare_inner(first, second, base)

    """
    Applies wu palmer similarity measure on two taxonomie elements
    """
    def compare_inner(self, first: str, second: str, base: Taxonomie) -> float:
        # Get directed graph
        d_graph = base.graph

        # Get undirected graph
        ud_graph = d_graph.to_undirected()

        # Get lowest reachable node from both        
        lowest_common_ancestor = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(d_graph, first, second)

        # Get root of graph
        root = [n for n,d in d_graph.in_degree() if d == 0][0]

        # Count edges - weight is 1 per default
        d1 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, first, lowest_common_ancestor)
        d2 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, second, lowest_common_ancestor)
        d3 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, lowest_common_ancestor, root)

        # if first and second, both is the root
        if d1 + d2 + 2 * d3 == 0.0:
            return 0.0

        return 2 * d3 / (d1 + d2 + 2 * d3)

    """
    Serializes a dict object with 2-tuples as key to json file
    """
    def __dump_json(self, dic, fileName) -> None:
        with open(fileName, "w") as f:
            k = dic.keys()
            v = dic.values()
            k1 = [str(i) for i in k]
            json.dump(json.dumps(dict(zip(*[k1,v]))),f)
    
    """
    Deserializes a json file to a dict object with 2-tuples as key
    """
    def __loads_json(self, fileName) -> dict:
        with open(fileName, "r") as f:
            data = json.load(f)
            dic = json.loads(data)
            k = dic.keys()
            v = dic.values()
            k1 = [eval(i) for i in k]
            return dict(zip(*[k1,v]))

    """
    Creates the cache for WuPalmer similarity, i.e. calculates pairwise values for
    taxonomy entries
    """
    def create_cache(self, base: Taxonomie) -> None:
        fileName = "cache/" + base.name + ".json"

        # check if cache already exist
        if os.path.isfile(fileName) and os.path.exists(fileName):
            Logger.debug("Cache for " + base.name + " already exist")
            return

        # format ((first taxonomy entry, second taxonomy entry), value)
        cache = dict()

        amount = int(math.pow(len(base.graph.nodes()), 2))
        index = 1
        everyNSteps = 100

        timer: Timer = Timer()
        timer.start()

        for first in base.graph.nodes():
            for second in base.graph.nodes():
                cache[(first, second)] = self.compare_inner(first, second, base)
                index += 1
                if index % everyNSteps == 0:
                    Logger.debug(str(index) + " from " + str(amount))

        if os.path.isdir("cache") == False:
            os.mkdir("cache")

        self.__dump_json(cache, fileName)

        timer.stop()

        return

"""
Represents a timecode comparer using the tanh function.
"""
class TimeTanh(ElementComparer):
    """
    Applies the tanh function for comparing timecodes.
    """
    def compare(self, first: int, second: int, base: Any) -> float:
        return np.tanh(np.abs( (first - second)) / 7200.0)
        