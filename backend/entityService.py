from typing import List
from typing import Tuple
import numpy as np
from backend.entity import Entity, EntityFactory
from backend.attribute import Attribute
from backend.attributeComparer import AttributeComparerType, AttributeComparerFactory
from backend.elementComparer import ElementComparerType, ElementComparerFactory
from backend.aggregator import AggregatorFactory, AggregatorType
from backend.transformer import TransformerFactory, TransformerType
from backend.database import Database
from backend.entityComparer import EntityComparer, EmptyAttributeAction
import os
import shutil
from backend.logger import Logger, LogLevel

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
        self.attributeComparerType = {}
        # format (attribute, elementComparerType)
        self.elementComparerType = {}
        # format (attribute, emptyAttributeAction)
        self.emptyAttributeAction = {}

        self.aggregatorType = None
        self.transformerType = None

        self.entitiyComparer = None

        # format (attribute, {alue1, value2, ...})
        self.filterRules = {}

        # all entities without filter
        self.allEntities = []

        # stores the filtered entities
        self.entities = []

        # store all the loaded domains, i.e. caching
        self.domains = {}

        return
    
    """
    Sets the seed for random return of entities.
    """
    def set_seed(self, value: int) -> None:
        np.random.seed(value)
        return

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
                self.elementComparerType[attribute] = elementComparerType
            if attributeComparerType is not None:
                self.attributeComparerType[attribute] = attributeComparerType
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
    Adds a filter rule to the list. A filter rule is tuple
    attribute and value and needs to be found in the dataset
    in order to work with it.
    """
    def add_filter_rule(self, attribute: Attribute, value: any) -> None:
        if attribute not in self.filterRules:
            self.filterRules[attribute] = {value}
        else:
            self.filterRules[attribute].add(value)
        return

    """
    Sets the aggregator that will be used to aggregate
    the attribute values.
    """
    def set_aggregator(self, aggregatorType: AggregatorType) -> None:
        self.aggregatorType = aggregatorType
        return

    """
    Sets the transformer to transform between similarities
    and distances.
    """
    def set_transformer(self, transformerType: TransformerType) -> None:
        self.transformerType = transformerType
        return

    """
    Creates the entities from the database based on the choosen attributes and amount.
    The default for amount is int max which returns all founded entities.
    """
    def create_entities(self, database: Database, amount: int = 2147483646) -> List[Entity]:
        self.allEntities = EntityFactory.create(self.attributes.keys(), database, amount)
        return

    """
    Creates the components, i.e. aggregator, transformer, attribute comparer and
    element comparer.
    """
    def create_components(self) -> None:
        self.entitiyComparer = EntityComparer(
            self.aggregatorType,
            self.transformerType
        )

        for attribute in self.elementComparerType.keys():
            self.entitiyComparer.add_element_comparer(
                attribute,
                self.elementComparerType[attribute]
            )

        for attribute in self.attributeComparerType.keys():
            self.entitiyComparer.add_attribute_comparer(
                attribute,
                self.attributeComparerType[attribute],
                self.emptyAttributeAction[attribute]
            )
        return

    """
    Forces all the elementComparer to build their caches, i.e. they make a pairwise
    comparsion of all elements in the taxonomy.
    """
    def create_caches(self) -> None:
        Logger.debug("Start creating caches")
        for attribute in self.entitiyComparer.elementComparer.keys():
            Logger.debug("Start caching " + Attribute.get_name(attribute))
            elementComparer = self.entitiyComparer.elementComparer[attribute]
            elementComparer.create_cache(Attribute.get_base(attribute))
        return

    """
    Deletes all the caches for the elementComparer
    """
    def delete_caches(self) -> None:
        if os.path.exists("cache") and os.path.isdir("cache"):
            shutil.rmtree("cache")
        return

    """
    Get all entities without doing filtering
    """
    def get_all_entities(self) -> List[Entity]:
        return self.allEntities

    """
    Gets the entities based on the choosen attributes and the filter.
    If there already exists filtered entities, those will be returned.
    """
    def get_entities(self) -> List[Entity]:
        if len(self.entities) > 0:
            return self.entities

        entities = self.allEntities

        for attribute in self.filterRules:
            for value in self.filterRules[attribute]:
                entities = list(filter(
                    lambda e: value in e.get_value(attribute),
                    entities))

        # teporarily disabled permutation
        #self.entities = np.random.permutation(entities)
        self.entities = entities

        return self.entities

    """
    Returns a set of all values that are in the loaded dataset
    for the given attribute
    """
    def get_domain(self, attribute: Attribute) -> List[any]:
        if attribute in self.domains:
            return self.domains[attribute]

        domain = set()
        for entity in self.allEntities:
            for value in entity.get_value(attribute):
                domain.add(value)
        
        self.domains[attribute] = domain
        return domain

    """
    Calculates the similarity of two given entity IDs
    """
    def calculate_similarity(self, id1, id2) -> float:
        # get the two entities out of the list
        entity1 = next((entity for entity in self.entities if entity.id == id1), None)
        entity2 = next((entity for entity in self.entities if entity.id == id2), None)

        # compare their values
        return self.entitiyComparer.calculate_similarity(entity1, entity2)

    """
    Calculates the distance of two given entity IDs
    """
    def calculate_distance(self, id1, id2) -> float:
        # get the two entities out of the list
        entity1 = next((entity for entity in self.entities if entity.id == id1), None)
        entity2 = next((entity for entity in self.entities if entity.id == id2), None)
        # compare their values
        return self.entitiyComparer.calculate_distance(entity1, entity2)
