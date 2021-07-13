"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
import networkx as nx
import os
import json
import math


class ElementComparerType(Enum):
    """
    Defines an enum to list up all available element comparer.
    """

    wuPalmer = 0


class ElementComparerFactory:
    """
    Represents the factory to create an element comparer
    """

    @classmethod
    def create(cls, element_comparer_type, base):
        """
        Creates an element comparer instance given the type.
        """

        if element_comparer_type == ElementComparerType.wuPalmer:
            return WuPalmer(base)
        else:
            raise Exception('Unknown type of element comparer')


class ElementComparer(metaclass=ABCMeta):
    """
    Represents the abstract element comparer base class.
    """

    def __init__(self, base):
        self._base = base

    @abstractmethod
    def compare(self, first, second):
        """
        Returns the comparison value of first and second
        element based on the given base.
        """
        pass

    @abstractmethod
    def create_cache(self, base):
        """
        Creates a full cache on file, i.e. calculates
        all pairwise similarities and safes the result
        in a file.
        """
        pass


class WuPalmer(ElementComparer):
    def __init__(self, base):
        super().__init__(base)
        self.__cache = None

    def compare(self, first, second):
        """
        Applies try to use cache first if available, if not,
        run compare_inner.
        """

        # check if cache is available
        cache = dict()
        if self.__cache is None:
            if os.path.isdir('cache'):
                file_name = 'cache/' + base.name + '.json'
                if os.path.isfile(file_name):
                    self.__cache = self.__load_json(file_name)
                    cache = self.__cache
        else:
            cache = self.__cache

        if (first, second) in cache:
            if (first, second) in cache:
                return self.__cache[(first, second)]

        return self.__compare_inner(first, second, base)

    def create_cache(self, base):
        """
        Creates the cache for WuPalmer similarity,
        i.e. calculates pairwise values for taxonomy entries.
        """

        file_name = 'cache/' + base.name + '.json'

        # check if cache already exist
        if os.path.isfile(file_name) and os.path.exists(file_name):
            return

        # format ((first taxonomy entry, second taxonomy entry), value)
        cache = dict()

        amount = int(math.pow(len(base.graph.nodes()), 2))
        index = 1
        every_n_steps = 100

        for first in base.graph.nodes():
            for second in base.graph.nodes():
                cache[(first, second)] = self.__compare_inner(first, second, base)
                index += 1
                if index % every_n_steps == 0:
                    Logger.debug(str(index) + ' from ' + str(amount))

        if not os.path.isdir('cache'):
            os.mkdir('cache')

        self.__dump_json(cache, file_name)

    @classmethod
    def __compare_inner(cls, first, second, base):
        """
        Applies wu palmer similarity measure on two taxonomy elements.
        """

        # Get directed graph
        d_graph = base.graph

        # Get undirected graph
        ud_graph = d_graph.to_undirected()

        # Get lowest reachable node from both
        lowest_common_ancestor = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(d_graph, first, second)

        # Get root of graph
        root = [n for n, d in d_graph.in_degree() if d == 0][0]

        # Count edges - weight is 1 per default
        d1 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, first, lowest_common_ancestor)
        d2 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, second, lowest_common_ancestor)
        d3 = nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, lowest_common_ancestor, root)

        # if first and second, both is the root
        if d1 + d2 + 2 * d3 == 0.0:
            return 0.0

        return 2 * d3 / (d1 + d2 + 2 * d3)

    @classmethod
    def __dump_json(cls, dic, file_name):
        """
        Serializes a dict object with 2-tuples as key to json file.
        """

        with open(file_name, 'w') as f:
            k = dic.keys()
            v = dic.values()
            k1 = [str(i) for i in k]
            json.dump(json.dumps(dict(zip(*[k1, v]))), f)

    @classmethod
    def __load_json(cls, file_name):
        """
        Deserializes a json file to a dict object with 2-tuples as key.
        """

        with open(file_name, 'r') as f:
            data = json.load(f)
            dic = json.loads(data)
            k = dic.keys()
            v = dic.values()
            k1 = [eval(i) for i in k]
            return dict(zip(*[k1, v]))
