from .elementComparer import ElementComparer
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
from backend.logger import Logger
from backend.attribute import Attribute

""" 
Defines an enum to list up all available attribute comparer.
"""
class AttributeComparerType(enum.Enum):
    symMaxMean = "symMaxMean"

    """
    Returns the name of a given AttributeComparerType.
    """
    @staticmethod
    def get_name(attributeComparerType) -> str:
        name = ""
        if attributeComparerType == AttributeComparerType.symMaxMean:
            name += "SymMaxMean"
        else:
            Logger.error("No name for attribute comparer \"" + str(attributeComparerType) + "\" specified")
            raise ValueError("No name for attribute comparer \"" + str(attributeComparerType) + "\" specified")
        return name

    """
    Returns the description to the given AttributeComparerType.
    """
    @staticmethod
    def get_description(attributeComparerType) -> str:
        description = ""
        if attributeComparerType == AttributeComparerType.symMaxMean:
            description += "Compares two attributes, i.e. sets of elements accordingly to " \
                + "setsim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b) " \
                + "+ 1/|B| sum_{b in B} max_{a in A} sim(b,a)) with sim() being " \
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
    Initializes the super attribute comparer class.
    """
    def __init__(self) -> None:
        # format of the cache:
        # ( (first, second), value )
        self.__cache = {}
        return

    """
    Adds the given similarity to the cache.
    Note that the chache is not commutative, i.e.
    if (firstElement, secondElement) is stored, there is
    no value for (secondElement, firstElement).
    This is, because some AttributeComparer may not symmetric.
    """
    def add_similarity_to_cache(self, firstElement: Any, secondElement: Any, similarity: float) -> None:
        self.__cache[(firstElement, secondElement)] = similarity

    """
    Returns True if the tuple of elements is already chached.
    """
    def is_similarity_in_cache(self, firstElement: Any, secondElement: Any) -> bool:
        return True if (firstElement, secondElement) in self.__cache else False

    """
    Loads the similarity between two elements from cache.
    """
    def get_similarity_from_cache(self, firstElement: Any, secondElement: Any) -> float:
        if (firstElement, secondElement) in self.__cache:
            return self.__cache[(firstElement, secondElement)]
        else:
            Logger.error(
                "Tried to load element similarity from cache but was not found!" + \
                "This comparsion will be done with 0.0 similarity"
                )
            return 0.0
    
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
        elementComparer: ElementComparer,
        attribute: Attribute
        ) -> AttributeComparer:
        if attributeComparerType == AttributeComparerType.symMaxMean:
            return SymMaxMean(elementComparer, Attribute.get_base(attribute))
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
    def __init__(self, elementComparer: ElementComparer, base: Any) -> None:
        super().__init__()
        self.elementComparer: ElementComparer = elementComparer
        self.base: Any = base
        return

    """
    Compares two attributes, i.e. sets of elements:
    setsim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b) 
                + 1/|B| sum_{b in B} max_{a in A} sim(b,a))
    """
    def compare(self, first: [Any], second: [Any]) -> float:
        sum1 = 0.0
        sum2 = 0.0

        # Sum over a in first
        for a in first:
            # Get maximum element_compare(a, b) with b in second
            max = 0.0
            for b in second:
                temp = 0.0

                # check if value is already in cache
                if self.is_similarity_in_cache(a, b):
                    temp = self.get_similarity_from_cache(a, b)
                else:
                    temp = self.elementComparer.compare(a, b, self.base)
                    self.add_similarity_to_cache(a, b, temp)

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
                temp = 0.0

                # check if value is already in cache
                if self.is_similarity_in_cache(b, a):
                    temp = self.get_similarity_from_cache(b, a)
                else:
                    temp = self.elementComparer.compare(b, a, self.base)
                    self.add_similarity_to_cache(b, a, temp)

                if temp > max:
                    max = temp
            
            sum2 += max

        # Normalize to size of second
        sum2 /= len(second)

        # Combine both sums
        return 0.5 * (sum1 + sum2)
