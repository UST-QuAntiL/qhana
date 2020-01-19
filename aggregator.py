from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum

""" 
Defines an enum to list up all available aggregator
"""
class AggregatorType(enum.Enum):
    mean = 1

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
Represents the factory to create element comparer
"""
class AggregatorFactory:

    @staticmethod
    def create(type: AggregatorType) -> AggregatorType:
        if type == AggregatorType.mean:
            return MeanAggregator()
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
