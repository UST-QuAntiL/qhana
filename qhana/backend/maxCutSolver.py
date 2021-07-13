from abc import ABCMeta, abstractmethod
from typing import List
import networkx as nx

"""
Represents an abstract MaxCutSolver class.
"""
class MaxCutSolver(metaclass=ABCMeta):

    """
    Instantiates the MaxCutSolver with the graph.
    """
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        return

    """
    Solves the max cut problem and returns the
    maximum cut in the format (cutValue, [(node1, node2), ...]),
    i.e. the cut value and the list of edges that
    correspond to the cut.
    """
    @abstractmethod
    def solve(self):
        return NotImplemented