from .elementComparer import ElementComparer
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum

""" 
Defines an enum to list up all available attribute comparer.
"""
class AttributeComparerType(enum.Enum):
    symMaxMean = 1
    singleElement = 2

    @staticmethod
    def get_description(attributeComparerType) -> str:
        description = ""
        if attributeComparerType == AttributeComparerType.symMaxMean:
            description += "Compares two attributes, i.e. sets of elements accordingly to " \
                + "setsim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b) " \
                + "+ 1/|B| sum_{b in B} max_{a in A} sim(b,a)) with sim() being " \
                + " the element comparer."
        elif attributeComparerType == AttributeComparerType.singleElement:
            description += "Compares two attributes which only consists of a single element. " \
                + "Therefore we apply setsim(A,B) = sim(A[0],B[0]) with sim() being " \
                + " the element comparer."
        else:
            Logger.error("No description for attribute comparer \"" + str(attributeComparerType) + "\" specified")
            raise ValueError("No description for attribute comparer \"" + str(attributeComparerType) + "\" specified")
        return description

""" 
Represents the abstract attribute comprarer base class.
"""
class AttributeComparer(metaclass=ABCMeta):
    """ 
    Returns the comparison value of first and second attribute.
    """
    @abstractmethod
    def compare(self, first: [Any], second: [Any], base: Any) -> float:
        return

""" 
Represents the factory to create an attribute comparer.
"""
class AttributeComparerFactory:
    """
    Abstract method for creating the attribute comparer.
    """
    @staticmethod
    def create(attributeComparerType: AttributeComparerType,
        elementComparer: ElementComparer
        ) -> AttributeComparer:
        if attributeComparerType == AttributeComparerType.symMaxMean:
            return SymMaxMean(elementComparer)
        elif attributeComparerType == AttributeComparerType.singleElement:
            return SingleElement(elementComparer)
        else:
            raise Exception("Unknown type of attribute comparer")

"""
Represents the attriute comparer with symmetrical observation, 
maximum element selection and averaging.
"""
class SymMaxMean(AttributeComparer):
    """
    Initializes the SymMaxMean attribute comparer.
    """
    def __init__(self, elementComparer: ElementComparer) -> None:
        super().__init__()
        self.elementComparer: ElementComparer = elementComparer
        return

    """
    Compares two attributes, i.e. sets of elements:
    setsim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b) 
                + 1/|B| sum_{b in B} max_{a in A} sim(b,a))
    """
    def compare(self, first: [Any], second: [Any], base: Any) -> float:
        sum1 = 0.0
        sum2 = 0.0

        # Sum over a in first
        for a in first:
            # Get maximum element_compare(a, b) with b in second
            max = 0.0
            for b in second:
                temp = self.elementComparer.compare(a, b, base)
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
                temp = self.elementComparer.compare(b, a, base)
                if temp > max:
                    max = temp
            
            sum2 += max

        # Normalize to size of second
        sum2 /= len(second)

        # Combine both sums
        return 0.5 * (sum1 + sum2)

"""
Represents the attriute comparer with attributes that are just one element. 
Therefore, just the element_comparer will be used.
"""
class SingleElement(AttributeComparer):
    """
    Initializes the SingleElement attribute comparer.
    """
    def __init__(self, elementComparer: ElementComparer) -> None:
        super().__init__()
        self.elementComparer: ElementComparer = elementComparer
        return

    """
    Compares two attributes, i.e. sets of elements, where each set has just one element:
    setsim(A,B) = sim(a,b).
    """
    def compare(self, first: [Any], second: [Any], base: Any) -> float:
        return self.elementComparer.compare(first[0], second[0], base)
