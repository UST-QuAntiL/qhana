from elementComparer import ElementComparer
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum

""" 
Defines an enum to list up all available attribute comparer
"""
class AttributeComparerType(enum.Enum):
    symMaxMean = 1

""" 
Defines an enum to specify what should be done
if an element is missing
"""
class MissingElementAction(enum.Enum):
    ignore = 1
    evaluate_as_zero = 2

""" 
Represents the abstract attribute comprarer base class
"""
class AttributeComparer(metaclass=ABCMeta):

    """ 
    Returns the comparison value of first and second attribute
    """
    @abstractmethod
    def compare(self, first: [Any], second: [Any], base: Any, *missingElementAction : MissingElementAction) -> float:
        return

""" 
Represents the factory to create element comparer
"""
class AttributeComparerFactory:

    @staticmethod
    def create(type: AttributeComparer, elementComparer: ElementComparer) -> AttributeComparer:
        if type == AttributeComparerType.symMaxMean:
            return SymMaxMean(elementComparer)
        else:
            raise Exception("Unknown type of attribute comparer")

"""
Represents the attriute comparer with symmetrical observation, 
maximum element selection and averaging
"""
class SymMaxMean(AttributeComparer):
    def __init__(self, elementComparer: ElementComparer) -> None:
        self.elementComparer: ElementComparer = elementComparer
        return

    """
    Compares two attributes, i.e. sets of elements:
    setsim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b) 
                + 1/|B| sum_{b in B} max_{a in A} sim(b,a))
    """
    def compare(self, base: Any, first: [Any], second: [Any]) -> float:
        sum1 = 0.0
        sum2 = 0.0

        # Sum over a in first
        for a in first:
            # Get maximum element_compare(a, b) with b in second
            max = 0.0
            for b in second:
                temp = self.elementComparer.compare(base, a, b)
                if temp > max:
                    max = temp
            
            sum1 += max
        
        # Normalize to size of first
        sum1 /= len(first)

        # Sum over b in second
        for b in second:
            # Get maximum element_compare(b, a) with a in first
            max = 0.0
            for a in first:
                temp = self.elementComparer.compare(base, b, a)
                if temp > max:
                    max = temp
            
            sum2 += max

        # Normalize to size of second
        sum2 /= len(second)

        # Combine both sums
        return 0.5 * (sum1 + sum2)
