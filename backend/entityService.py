from typing import List
from typing import Tuple
from backend.entity import Entity, EntityFactory
from backend.attribute import Attribute
from backend.attributeComparer import AttributeComparerType, AttributeComparerFactory
from backend.elementComparer import ElementComparerType, ElementComparerFactory
from backend.aggregator import AggregatorFactory, AggregatorType
from backend.transformer import TransformerFactory, TransformerType
from backend.database import Database
from backend.entityComparer import EntityComparer, EmptyAttributeAction

"""
This class is a searvice for the easy construction 
and use of entities and their comparer.
"""
class EntityService:
    """
    Initializes the EntityService.
    """
    def __init__(self) -> None:
        # format (attribute, None)
        self.attributes = {}
        # format (attribute, attributeComparerType)
        self.attributeComparer = {}
        # format (attribute, elementComparerType)
        self.elementComparer = {}
        # format (attribute, emptyAttributeAction)
        self.emptyAttributeAction = {}

        self.aggregator = None
        self.transformer = None
    
    """
    Adds an attribute with the corresponding
    element and attribute comparer. If element
    or attribute comparer is none, the corresponding
    attribute will be a tag.
    """
    def add_attribute(self,
        attribute: Attribute,
        elementComparerType: ElementComparerType = None,
        attributeComparerType: AttributeComparerType = None,
        emptyAttributeAction: EmptyAttributeAction = EmptyAttributeAction.ignore
        ) -> None:
            self.attributes[attribute] = None
            if elementComparerType is not None:
                elementComparer = ElementComparerFactory.create(elementComparerType)
                self.elementComparer[attribute] = elementComparer
            if attributeComparerType is not None:
                attributeComparer = AttributeComparerFactory.create(attributeComparerType, elementComparer)
                self.attributeComparer[attribute] = attributeComparer
            self.emptyAttributeAction[attribute] = emptyAttributeAction
            return

    """
    Adds a plan to the list. A plan is a list with specifying
    an attribute, the corresponding element comparer and
    attribute comparer as well as the emptyAttributeAction.
    """
    def add_plan(self, plan) -> None:
        for element in plan:
            if isinstance(element, AggregatorType):
                self.set_aggregator(element)
            elif isinstance(element, TransformerType):
                self.set_transformer(element)
            else:
                self.add_attribute(element[0], element[1], element[2], element[3])
        return

    """
    Sets the aggregator that will be used to aggregate
    the attribute values.
    """
    def set_aggregator(self, aggregator: AggregatorType) -> None:
        self.aggregator = AggregatorFactory.create(aggregator)
        return

    """
    Sets the transformer to transform between similarities
    and distances.
    """
    def set_transformer(self, transformer: TransformerType) -> None:
        self.transformer = TransformerFactory.create(transformer)
        return

    """
    Gets the entities based on the choosen attributes, database and amount.
    The default for amount is int max which returns all founded entities.
    """
    def get_entities(self, database: Database, amount: int = 2147483646) -> List[Entity]:
        entities = EntityFactory.create(self.attributes.keys(), database, amount)
        return entities
