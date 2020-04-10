from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import networkx as nx
from networkx import Graph
from backend.taxonomie import Taxonomie

""" 
Defines an enum to list up all available element comparer
"""
class ElementComparerType(enum.Enum):
    wuPalmer = 1

    @staticmethod
    def get_description(elementComparerType) -> str:
        description = ""
        if elementComparerType == ElementComparerType.wuPalmer:
            description += "Compares two elements based on a taxonomie " \
                + "using the wu palmer similarity measure."
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
        else:
            raise Exception("Unknown type of element comparer")
        return

"""
Represents the conventional wu palmer similarity measure
"""
class WuPalmer(ElementComparer):
    """
    Applies wu palmer similaritie measure on two taxonomie elements
    """
    def compare(self, first: str, second: str, base: Taxonomie) -> float:
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

        return 2 * d3 / (d1 + d2 + 2* d3)
        