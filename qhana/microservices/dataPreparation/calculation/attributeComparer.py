"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import ABCMeta
from abc import abstractmethod
import enum


class AttributeComparerType(enum.Enum):
    """
    Defines an enum to list up all available attribute comparer.
    """
    symMaxMean = 0


class AttributeComparerFactory:
    """
    Represents the factory to create an attribute comparer.
    """

    @classmethod
    def create(cls, attribute_comparer_type, element_comparer, base):
        """
        Abstract method for creating the attribute comparer.
        """

        if attribute_comparer_type == AttributeComparerType.symMaxMean:
            return SymMaxMean(element_comparer, base)
        else:
            raise Exception('Unknown type of attribute comparer.')


class AttributeComparer(metaclass=ABCMeta):
    """
    Represents the abstract attribute comparer base class.
    """

    def __init__(self, base) -> None:
        # format of the cache:
        # [(first, second), value]
        self.__cache = dict()
        self._base = base

    def _add_similarity_to_cache(self, first_element, second_element, similarity):
        """
        Adds the given similarity to the cache.
        Note that the cache is not commutative, i.e.
        if (first, second) is stored, there is
        no value for (second, first).
        This is, because some AttributeComparer may not symmetric.
        """
        self.__cache[(first_element, second_element)] = similarity

    def _is_similarity_in_cache(self, first_element, second_element):
        """
        Returns True if the tuple of elements is already cached.
        """
        return (first_element, second_element) in self.__cache

    def _get_similarity_from_cache(self, first_element, second_element):
        """
        Loads the similarity between two elements from cache. If the
        element is not in the cache, we return 0.
        """
        if self._is_similarity_in_cache(first_element, second_element):
            return self.__cache[(first_element, second_element)]
        else:
            return 0.0

    @abstractmethod
    def compare(self, first, second):
        """
        Returns the comparison value of first and second attribute.
        """
        pass


class SymMaxMean(AttributeComparer):
    """
    Represents the attribute comparer with symmetrical observation,
    maximum element selection and averaging.
    """

    def __init__(self, element_comparer, base):
        super().__init__(base)
        self.__element_comparer = element_comparer

    def compare(self, first, second):
        """
        Compares two attributes, i.e. sets of elements:
        set_sim(A,B) = 1/2 * (1/|A| sum_{a in A} max_{b in B} sim(a,b)
                    + 1/|B| sum_{b in B} max_{a in A} sim(b,a))
        """

        sum1 = 0.0
        sum2 = 0.0

        # Sum over a in first
        for a in first:
            # Get maximum element_compare(a, b) with b in second
            max_value = 0.0
            for b in second:
                # check if value is already in cache
                if self._is_similarity_in_cache(a, b):
                    temp = self._get_similarity_from_cache(a, b)
                else:
                    temp = self.__element_comparer.compare(a, b, self._base)
                    self._add_similarity_to_cache(a, b, temp)

                if temp > max_value:
                    max_value = temp

            sum1 += max_value

        # Normalize to size of first
        sum1 /= len(first)

        # Sum over b in second
        for b in second:
            # Get maximum element_compare(b, a) with a in first
            max_value = 0.0
            for a in first:
                # check if value is already in cache
                if self._is_similarity_in_cache(b, a):
                    temp = self._get_similarity_from_cache(b, a)
                else:
                    temp = self.__element_comparer.compare(b, a, self._base)
                    self._add_similarity_to_cache(b, a, temp)

                if temp > max_value:
                    max_value = temp

            sum2 += max_value

        # Normalize to size of second
        sum2 /= len(second)

        # Combine both sums
        return 0.5 * (sum1 + sum2)
