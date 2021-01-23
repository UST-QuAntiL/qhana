"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import enum
import numpy as np
from aggregator import AggregatorType, AggregatorFactory
from transformer import TransformerType, TransformerFactory
from elementComparer import ElementComparerType, ElementComparerFactory
from attributeComparer import AttributeComparerType, AttributeComparerFactory


class EmptyAttributeAction(enum.Enum):
    """
    Defines an enum to specify what should be done
    if an attribute is missing.
    """

    ignore = 0
    evaluateAsZero = 0


# noinspection PyBroadException
class EntityComparer:
    """
    Represents the comparer class for arbitrary entities.
    """

    def __init__(self, aggregator_type=AggregatorType.mean, transformer_type=TransformerType.squareInverse):
        # stores all the registered element comparer in the format
        # [attribute_name, element_comparer]
        self.__element_comparer = dict()

        # stores all the registered attribute comparer in the format
        # [attribute_name, (attribute_comparer, empty_attribute_action)]
        self.__attribute_comparer = dict()

        # creates the comparer and aggregator with the factory pattern
        self.__attribute_aggregator = AggregatorFactory.create(aggregator_type)
        self.__similarity_transformer = TransformerFactory.create(transformer_type)

        # format of the cache:
        # [ ([first], [second]), value ]
        # with the lists being converted to tuples in order
        # to be hashable
        self.__cache = dict()

    def __add_similarity_to_cache(self, first, second, similarity):
        """
        Adds the given similarity to the cache.
        Note that the cache is not commutative, i.e.
        if ([first], [second]) is stored, there is
        no value for ([second], [first]).
        This is, because some attribute comparer may not symmetric.
        """

        self.__cache[(tuple(first), tuple(second))] = similarity

    def __is_similarity_in_cache(self, first, second):
        """
        Returns True if the tuple of attributes is already cached.
        """

        return (tuple(first), tuple(second)) in self.__cache

    def __get_similarity_from_cache(self, first, second) -> float:
        """
        Loads the similarity between two attributes from cache.
        If the element is not in the cache, return 0.
        """

        if self.__is_similarity_in_cache(first, second):
            return self.__cache[(tuple(first), tuple(second))]
        else:
            return 0.0

    def add_element_comparer(self, attribute, element_comparer_type=ElementComparerType.wuPalmer):
        """
        Registers an element comparer that will be used
        to compare all the elements within the attribute
        that contains these elements.
        """

        element_comparer = ElementComparerFactory.create(element_comparer_type)
        self.__element_comparer[attribute] = element_comparer

    def add_attribute_comparer(self,
                               attribute,
                               base,
                               attribute_comparer_type=AttributeComparerType.symMaxMean,
                               empty_attribute_action=EmptyAttributeAction.ignore):
        """
        Registers an attribute comparer that will be used
        to compare all these attributes within the entities.
        If we cannot find an element comparer, we will take the default.
        """

        # check if element comparer is already registered.
        if attribute not in self.__element_comparer:
            self.add_element_comparer(attribute)

        element_comparer = self.__element_comparer[attribute]
        attribute_comparer = AttributeComparerFactory.create(attribute_comparer_type, element_comparer, attribute)
        self.__attribute_comparer[attribute] = (attribute_comparer, empty_attribute_action)

    def calculate_similarity(self, first, second):
        """
        Returns the similarity between the two given entities based on
        the registered attribute compares.
        If there is no attribute comparer, the attribute will just use
        as a tag and do not influence the comparing result.
        """

        aggregation_values = []

        # iterate through all attributes form first entity and compare
        for attribute in first.attributes:
            # check if an attribute comparer is registered
            # for the given attribute. if not, just skip
            # as the attribute is just a tag.
            if attribute not in self.__attribute_comparer:
                continue

            attribute_comparer = self.__attribute_comparer[attribute][0]
            empty_attribute_action = self.__attribute_comparer[attribute][1]

            # check if second entity has the same attribute
            if attribute not in second.attributes:
                # act according to emptyAttributeAction
                if empty_attribute_action == EmptyAttributeAction.ignore:
                    continue
                elif empty_attribute_action == EmptyAttributeAction.evaluateAsZero:
                    aggregation_values.append(0.0)
                    continue
                else:
                    # if unknown empty_attribute_action, ignore it.
                    continue

            first_attribute = first.values[attribute]
            second_attribute = second.values[attribute]

            # check if the attributes has values
            if len(first_attribute) == 0 or len(second_attribute) == 0:
                # act according to emptyAttributeAction
                if empty_attribute_action == EmptyAttributeAction.ignore:
                    continue
                elif empty_attribute_action == EmptyAttributeAction.evaluateAsZero:
                    aggregation_values.append(0.0)
                    continue
                else:
                    # if unknown empty_attribute_action, ignore it.
                    continue

            # check if value is already in cache
            if self.__is_similarity_in_cache(first_attribute, second_attribute):
                aggregation_values.append(self.__get_similarity_from_cache(first_attribute, second_attribute))
            else:
                # compare the attributes
                try:
                    temp = attribute_comparer.compare(first_attribute, second_attribute)
                    self.__add_similarity_to_cache(first_attribute, second_attribute, temp)
                    aggregation_values.append(temp)
                except:
                    err_message = 'Exception in comparison. '
                    if empty_attribute_action == EmptyAttributeAction.ignore:
                        err_message = err_message + ' The attribute will be ignored.'
                        print(err_message)
                        continue
                    elif empty_attribute_action == EmptyAttributeAction.evaluateAsZero:
                        err_message = err_message + ' The attribute will be evaluated as zero.'
                        print(err_message)
                        aggregation_values.append(0.0)
                        continue
                    else:
                        # if unknown empty_attribute_action, ignore it.
                        continue

        # check if there are attributes from the second entity
        # that are not part of the first entity
        for attribute in second.attributes:
            if attribute not in first.attributes:
                # check if a attribute comparer is registered
                # for the given attribute. if not, just skip
                # as the attribute is just a tag.
                if attribute not in self.__attribute_comparer:
                    continue

                empty_attribute_action = self.__attribute_comparer[attribute][1]

                if empty_attribute_action == EmptyAttributeAction.ignore:
                    continue
                elif empty_attribute_action == EmptyAttributeAction.evaluateAsZero:
                    aggregation_values.append(0.0)
                    continue
                else:
                    # if unknown empty_attribute_action, ignore it.
                    continue

        return self.__attribute_aggregator.aggregate(aggregation_values)

    def calculate_pairwise_similarity(self, entities):
        """
        Calculates the pairwise similarity and returns the similarity matrix.
        """

        result = np.array((len(entities), len(entities)))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                result[i][j] = self.calculate_similarity(entities[i], entities[j])

    def calculate_distance(self, first, second) -> float:
        """
        Returns the distance between the two given entities based on the
        defined similarity transformer and attribute comparer.
        """
        return self.__similarity_transformer.transform(self.calculate_similarity(first, second))

    def calculate_pairwise_distance(self, entities):
        """
        Calculates the pairwise distance and returns the distance matrix.
        """

        result = np.array((len(entities), len(entities)))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                result[i][j] = self.calculate_distance(entities[i], entities[j])
