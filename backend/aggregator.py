from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum

""" 
Defines an enum to list up all available aggregator
"""
class AggregatorType(enum.Enum):
    mean = 1
    median = 2
    max = 3
    min = 4

""" 
Represents the abstract aggregator base class
"""
class Aggregator(metaclass=ABCMeta):
    """ 
    Returns the aggregated value of the given list of values
    """
    @abstractmethod
    def aggregate(self, values: [float]) -> float:
        pass

""" 
Represents the factory to create an aggregator
"""
class AggregatorFactory:
    """
    Static method for creating the aggregator
    """
    @staticmethod
    def create(type: AggregatorType) -> AggregatorType:
        if type == AggregatorType.mean:
            return MeanAggregator()
        elif type == AggregatorType.median:
            return MedianAggregator()
        elif type == AggregatorType.max:
            return MaxAggregator()
        elif type == AggregatorType.min:
            return MinAggregator()
        else:
            raise Exception("Unknown type of aggregator")

"""
Represents the mean aggregator
"""
class MeanAggregator(Aggregator):
    """ 
    Returns the mean value of the given list of values
    """
    def aggregate(self, values: [float]) -> float:
        result = 0.0

        for value in values:
            result += value
        
        return result / len(values)

"""
Represents the median aggregator
"""
class MedianAggregator(Aggregator):
    """ 
    Returns the median of the given list of values
    """
    def aggregate(self, values: [float]) -> float:
        result = 0.0

        sortedList = values.copy()
        sortedList.sort()

        # Calculate median: if even number of elements take mean value
        if len(sortedList) % 2 == 0:
            result = 0.5 * (sortedList[len(sortedList / 2)] + sortedList[len(sortedList / 2 - 1)])
        else:
            result = sortedList[len(sortedList / 2)]
        
        return result

"""
Represents the max aggregator
"""
class MaxAggregator(Aggregator):
    """ 
    Returns the maximum of the given list of values
    """
    def aggregate(self, values: [float]) -> float:
        return max(values)

"""
Represents the min aggregator
"""
class MinAggregator(Aggregator):
    """ 
    Returns the minimum of the given list of values
    """
    def aggregate(self, values: [float]) -> float:
        return min(values)
