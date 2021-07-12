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
import enum

Subset5PositiveFileName = "./subsets/5/PositiveSubset5.csv"
Subset5NegativeFileName = "./subsets/5/NegativeSubset5.csv"
Subset10PositiveFileName = "./subsets/10/PositiveSubset10.csv"
Subset10NegativeFileName = "./subsets/10/NegativeSubset10.csv"
Subset25PositiveFileName = "./subsets/25/PositiveSubset25.csv"
Subset25NegativeFileName = "./subsets/25/NegativeSubset25.csv"
Subset40PositiveFileName = "./subsets/40/PositiveSubset40.csv"
Subset40NegativeFileName = "./subsets/40/NegativeSubset40.csv"

class Subset(enum.Enum):
    subset5 = "Subset5",
    subset10 = "Subset10",
    subset25 = "Subset25",
    subset40 = "Subset40"

    @staticmethod
    def get_name(subset) -> str:
        if subset == Subset.subset5:
            return "Subset5"
        elif subset == Subset.subset10:
            return "Subset10"
        elif subset == Subset.subset25:
            return "Subset25"
        elif subset == Subset.subset40:
            return "Subset40"
        else:
            Logger.error("No name for subset \"" + str(subset) + "\" specified")
            raise ValueError("No name for subset \"" + str(subset) + "\" specified")
        return

    @staticmethod
    def get_subset(subsetString) -> str:
        if subsetString == "Subset5":
            return Subset.subset5
        elif subsetString == "Subset10":
            return Subset.subset10
        elif subsetString == "Subset25":
            return Subset.subset25
        elif subsetString == "Subset40":
            return Subset.subset40
        else:
            Logger.error("No subset for name \"" + str(subsetString) + "\" specified")
            raise ValueError("No subset for name \"" + str(subsetString) + "\" specified")
        return

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
    def add_attribute(
        self,
        attribute: Attribute,
        elementComparerType: ElementComparerType = None,
        attributeComparerType: AttributeComparerType = None,
        emptyAttributeAction: EmptyAttributeAction = EmptyAttributeAction.ignore,
        filter_str: str = ""
        ) -> None:
            self.attributes[attribute] = None
            if elementComparerType is not None:
                self.elementComparerType[attribute] = elementComparerType
            if attributeComparerType is not None:
                self.attributeComparerType[attribute] = attributeComparerType
            self.emptyAttributeAction[attribute] = emptyAttributeAction
            self.filterRules[attribute] = filter_str.split(",")

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
                self.add_attribute(element[0], element[1], element[2], element[3], element[4])
        return

    """
    Adds a filter rule to the list. A filter rule is tuple
    attribute and value and needs to be found in the dataset
    in order to work with it.
    """
    # TODO: reuse or replace?
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

    def create_subset_from_file(self, database: Database, positiveCsvFile: str, negativeCsvFile: str):
        kostuemIDs = []
        rollenIDs = []
        filmIDs = []

        with open(positiveCsvFile, 'r') as f:
            for row in f:
                kostuemId = row.split(',')[1]
                if kostuemId.isdigit():
                    kostuemIDs.append(row.split(',')[1])
                    rollenIDs.append(row.split(',')[2])
                    filmIDs.append(row.split(',')[3])

        with open(negativeCsvFile, 'r') as f:
            for row in f:
                kostuemId = row.split(',')[1]
                if kostuemId.isdigit():
                    kostuemIDs.append(row.split(',')[1])
                    rollenIDs.append(row.split(',')[2])
                    filmIDs.append(row.split(',')[3])

        keys = []
        for i in range(0, len(kostuemIDs)):
            keys.append((kostuemIDs[i], rollenIDs[i], filmIDs[i]))

        unsortedEntities = EntityFactory.create(self.attributes.keys(), database, len(kostuemIDs), keys)
        sortedEntities = []
        # sort the subset accordingly to the csv file
        for i in range(0, len(kostuemIDs)):
            elem = self.get_costume_with(unsortedEntities, keys[i][0], keys[i][1], keys[i][2])
            sortedEntities.append(elem)
            #if elem is None:
            #    print("ERROR!")
            #print("kostuem = " + str(keys[i][0]) + " rolle = " + str(keys[i][1]) + " film = " + str(keys[i][2]))

        for i in range(0, len(sortedEntities)):
            sortedEntities[i].set_id(i)

        self.allEntities = sortedEntities

    def get_costume_with(self, costumes, kostuemId, rollenId, filmId):
        for elem in costumes:
            if str(elem.kostuemId) == kostuemId and str(elem.rollenId) == rollenId and str(elem.filmId == filmId):
                return elem
        return None

    def create_subset(self, subsetEnum: Subset, database: Database) -> List[Entity]:
        if subsetEnum == Subset.subset5:
            self.create_subset_from_file(database, Subset5PositiveFileName, Subset5NegativeFileName)
        elif subsetEnum == Subset.subset10:
            self.create_subset_from_file(database, Subset10PositiveFileName, Subset10NegativeFileName)
        elif subsetEnum == Subset.subset25:
            self.create_subset_from_file(database, Subset25PositiveFileName, Subset25NegativeFileName)
        elif subsetEnum == Subset.subset40:
            self.create_subset_from_file(database, Subset40PositiveFileName, Subset40NegativeFileName)
        else:
            Logger.error("You took a wrong value for subset.")


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
    Gets the entities based on the chosen attributes and the filter.
    If there already exists filtered entities, those will be returned.
    """
    def get_entities(self, useRandom: bool = False) -> List[Entity]:
        if len(self.entities) > 0:
            return self.entities

        entities = self.allEntities

        def is_value_in_list_of_values(v: str, lv: List[str]) -> bool:
            for v2 in lv:
                if v.lower() == v2.lower():
                    return True

            return False

        for attribute in self.filterRules:
            value_filtered_entities = set()

            for value in self.filterRules[attribute]:
                value_filtered_entities = list(set(filter(
                    lambda e: is_value_in_list_of_values(value, e.get_value(attribute)),  # TODO: extend with infos from the taxonomy, allow more complicated filter expressions
                    entities)).union(value_filtered_entities))

            entities = list(value_filtered_entities)

        # teporarily disabled permutation
        if useRandom:
            self.entities = np.random.permutation(entities)
        else:
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
