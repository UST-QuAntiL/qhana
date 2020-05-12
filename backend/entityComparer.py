from backend.elementComparer import ElementComparerType, ElementComparer, ElementComparerFactory
from backend.attributeComparer import AttributeComparerType, AttributeComparer, AttributeComparerFactory
from backend.transformer import TransformerType, TransformerFactory
from backend.aggregator import AggregatorType, AggregatorFactory
import enum
from backend.attribute import Attribute
from backend.entity import Entity
from backend.logger import Logger
from typing import Any

""" 
Defines an enum to specify what should be done
if an attribute is missing
"""
class EmptyAttributeAction(enum.Enum):
    ignore = "ignore"
    evaluateAsZero = "evaluateAsZero"

    """
    Returns the name of the EmptyAttributeAction.
    """
    @staticmethod
    def get_name(variable) -> str:
        if variable == EmptyAttributeAction.ignore:
            return "ignore"
        elif variable == EmptyAttributeAction.evaluateAsZero:
            return "evaluate as zero"
        else:
            Logger.error("No name for EmptyAttributeAction \"" + str(variable) + "\" specified")
            raise ValueError("No name for EmptyAttributeAction \"" + str(variable) + "\" specified")
        return

    """
    Returns the EmptyAttributeActionType given to the name
    """
    @staticmethod
    def get_emptyAttributeAction_type(variable) -> str:
        if variable == EmptyAttributeAction.ignore:
            return EmptyAttributeAction.ignore
        elif variable == EmptyAttributeAction.evaluateAsZero:
            return EmptyAttributeAction.evaluateAsZero
        else:
            Logger.error("No Type for EmptyAttributeAction \"" + str(variable) + "\" specified")
            raise ValueError("No Type for EmptyAttributeAction \"" + str(variable) + "\" specified")
        return

"""
Represents the comparer class for arbitrary entities
"""
class EntityComparer:
    """
    Initializes the entity comparer
    """
    def __init__(
        self,
        attributeAggregatorType: AggregatorType = AggregatorType.mean,
        similarityTransformerType: TransformerType = TransformerType.linearInverse
        ) -> None:

        # stores all the registered element comparer in the format
        # (name, element_comparer)
        self.elementComparer = {}

        # stores all the registered attribute comparer in the format
        # (name, (attribute_comparer, emptyAttributeAction))
        self.attributeComparer = {}

        # Creates the comparer and aggregator due to factory pattern
        self.attributeAggregator = AggregatorFactory.create(attributeAggregatorType)
        self.similarityTransformer = TransformerFactory.create(similarityTransformerType)

        # format of the cache:
        # ( ([first], [second]), value )
        # while the lists are converted to tuples in order
        # to be hashable
        self.__cache = {}
        
        return

    """
    Adds the given similarity to the cache.
    Note that the chache is not commutative, i.e.
    if ([first], [second]) is stored, there is
    no value for ([second], [first]).
    This is, because some AttributeComparer may not symmetric.
    """
    def __add_similarity_to_cache(self, first: [Any], second: [Any], similarity: float) -> None:
        self.__cache[(tuple(first), tuple(second))] = similarity

    """
    Returns True if the tuple of attributes is already chached.
    """
    def __is_similarity_in_cache(self, first: [Any], second: [Any]) -> bool:
        return True if (tuple(first), tuple(second)) in self.__cache else False

    """
    Loads the similarity between two attributes from cache.
    """
    def __get_similarity_from_cache(self, first: [Any], second: [Any]) -> float:
        if (tuple(first), tuple(second)) in self.__cache:
            return self.__cache[(tuple(first), tuple(second))]
        else:
            Logger.error(
                "Tried to load attribute similarity from cache but was not found!" + \
                "This comparsion will be done with 0.0 similarity"
                )
            return 0.0

    """
    Registers an element comparer that will be used
    to compare all the elements within the attribute
    that contains these elements
    """
    def add_element_comparer(self,
        attribute: Attribute,
        elementComparerType: ElementComparerType
        ) -> None:

        elementComparer = ElementComparerFactory.create(elementComparerType)
        self.elementComparer[attribute] = elementComparer
        return

    """
    Registers an attribute comparer that will be used
    to compare all the attributes within the entities
    """
    def add_attribute_comparer(self,
        attribute: Attribute,
        attributeComparerType: AttributeComparerType,
        emptyAttributeAction: EmptyAttributeAction
        ) -> None:

        # check if element comparer is already registered
        if attribute not in self.elementComparer:
            Logger.error("For the attribute \"" + 
                Attribute.get_name(attribute) + "\" " + 
                "is no element comparer registered."
            )
            return

        elementComparer = self.elementComparer[attribute]
        attributeComparer = AttributeComparerFactory.create(attributeComparerType, elementComparer)
        self.attributeComparer[attribute] = (attributeComparer, emptyAttributeAction)
        return

    """
    Returns the similarity between the two given entities based on
    the registered attribute compares.
    If there is no attribute comparer, the attribute will just use
    as a tag and do not influence the comparing result
    """
    def calculate_similarity(self, first: Entity, second: Entity) -> float:
        aggregationValues = []

        # iterate through all attributes form first entity
        # and compare
        for attribute in first.attributes:
            # check if a attribute comparer is registered
            # for the given attribute. if not, just skip
            # as the attribute is just a tag.
            if attribute not in self.attributeComparer:
                continue

            attributeComparer = self.attributeComparer[attribute][0]
            emptyAttributeAction = self.attributeComparer[attribute][1]

            # check if second entity has the same attribute
            if attribute not in second.attributes:
                # act according to emptyAttributeAction
                if emptyAttributeAction == EmptyAttributeAction.ignore:
                    continue
                elif emptyAttributeAction == EmptyAttributeAction.evaluateAsZero:
                    return 0.0
                else:
                    Logger.warning("Unknown value for EmptyAttributeAction. This attribute will be ignored.")
                    continue

            firstAttribute = first.values[attribute]
            secondAttribute = second.values[attribute]

            # check if value is already in chace
            if self.__is_similarity_in_cache(firstAttribute, secondAttribute):
                aggregationValues.append(
                    self.__get_similarity_from_cache(firstAttribute, secondAttribute)
                )
            else:
                # compare the attributes
                temp = attributeComparer.compare(
                    firstAttribute,
                    secondAttribute,
                    first.bases[attribute]
                )
                self.__add_similarity_to_cache(firstAttribute, secondAttribute, temp)
                aggregationValues.append(temp)


        # check if there are attributes from the second entity
        # that are not part of the first entity
        for attribute in second.attributes:
            if attribute not in first.attributes:
                # check if a attribute comparer is registered
                # for the given attribute. if not, just skip
                # as the attribute is just a tag.
                if attribute not in self.attributeComparer:
                    continue

                emptyAttributeAction = self.attributeComparer[attribute][1]

                if emptyAttributeAction == EmptyAttributeAction.ignore:
                    continue
                elif emptyAttributeAction == EmptyAttributeAction.evaluateAsZero:
                    return 0.0
                else:
                    Logger.warning("Unknown value for EmptyAttributeAction. This attribute will be ignored.")
                    continue

        return self.attributeAggregator.aggregate(aggregationValues)

    """
    Returns the distance between the two given entities based on the 
    defined similarity transformer and attribute comparer
    """
    def calculate_distance(self, first: Entity, second: Entity) -> float:
        return self.similarityTransformer.transform(self.calculate_similarity(first, second))

"""
Represents the comparer class for costumes
"""
class CostumeComparer(EntityComparer):
    """
    Initializes the costume comparer
    """
    def __init__(
        self,
        attributeAggregatorType: AggregatorType = AggregatorType.mean,
        similarityTransformerType: TransformerType = TransformerType.linearInverse) -> None:

        super().__init__(attributeAggregatorType, similarityTransformerType)

        # register all the element comparer for the costume comparing
        # this needs to be done first
        # dominanteFarbe
        self.add_element_comparer(
            Attribute.dominanteFarbe,
            ElementComparerType.wuPalmer
        )
        # dominanteCharactereigenschaft
        self.add_element_comparer(
            Attribute.dominanteCharaktereigenschaft,
            ElementComparerType.wuPalmer
        )
        # dominanterZustand
        self.add_element_comparer(
            Attribute.dominanterZustand,
            ElementComparerType.wuPalmer
        )
        # stereotyp
        self.add_element_comparer(
            Attribute.stereotyp,
            ElementComparerType.wuPalmer,
        )
        # geschlecht
        self.add_element_comparer(
            Attribute.geschlecht,
            ElementComparerType.wuPalmer,
        )
        # dominanterAlterseindruck
        self.add_element_comparer(
            Attribute.dominanterAlterseindruck,
            ElementComparerType.wuPalmer,
        )
        # genre
        # do nothing - just a tag

        # register all the attribute comparer for the costume comparing
        # dominanteFarbe
        self.add_attribute_comparer(
            Attribute.dominanteFarbe,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # dominanteCharaktereigenschaft
        self.add_attribute_comparer(
            Attribute.dominanteCharaktereigenschaft,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # dominanterZustand
        self.add_attribute_comparer(
            Attribute.dominanterZustand,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # stereotyp
        self.add_attribute_comparer(
            Attribute.stereotyp,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # geschlecht
        self.add_attribute_comparer(
            Attribute.geschlecht,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # dominanterAlterseindruck
        self.add_attribute_comparer(
            Attribute.dominanterAlterseindruck,
            AttributeComparerType.symMaxMean,
            EmptyAttributeAction.ignore
        )
        # genre
        # do nothing - just a tag

        return
