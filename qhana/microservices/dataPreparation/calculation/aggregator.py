"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import ABCMeta
from abc import abstractmethod
from enum import Enum


class AggregatorType(Enum):
    """
    Defines an enum to list up all available aggregators.
    """

    mean = 0
    median = 1
    max = 2
    min = 3


class AggregatorFactory:
    """
    Represents the factory to create a aggregators.
    """

    @classmethod
    def create(cls, aggregator_type):
        """
        Create an aggregator instance given the type.
        """

        if aggregator_type == AggregatorType.mean:
            return MeanAggregator()
        elif aggregator_type == AggregatorType.median:
            return MedianAggregator()
        elif aggregator_type == AggregatorType.max:
            return MaxAggregator()
        elif aggregator_type == AggregatorType.min:
            return MinAggregator()
        else:
            raise Exception("Unknown type of aggregator")


class Aggregator(metaclass=ABCMeta):
    """
    Represents the abstract aggregator base class.
    """

    @abstractmethod
    def aggregate(self, values):
        """
        Returns the aggregated value of the given list of values.
        """
        pass


class MeanAggregator(Aggregator):
    def aggregate(self, values):
        result = 0.0
        for value in values:
            result += value
        return result / len(values)


class MedianAggregator(Aggregator):
    def aggregate(self, values):
        sorted_list = values.copy()
        sorted_list.sort()
        # Calculate median: if even number of elements take mean value
        if len(sorted_list) % 2 == 0:
            return 0.5 * (sorted_list[len(sorted_list / 2)] + sorted_list[len(sorted_list / 2 - 1)])
        else:
            return sorted_list[len(sorted_list / 2)]


class MaxAggregator(Aggregator):
    def aggregate(self, values):
        return max(values)


class MinAggregator(Aggregator):
    def aggregate(self, values):
        return min(values)
