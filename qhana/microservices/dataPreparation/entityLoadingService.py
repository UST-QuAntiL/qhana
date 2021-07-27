"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import csv
from entity import Entity


class EntityLoadingService:
    """
    A service class to load entities from csv files.
    """

    @classmethod
    def load_entities(cls, file_path):
        """
        Load all entities from the given csv file.
        We always load all attributes we find in the file.
        We return a list of entity objects.
        """

        entities = []
        with open(file_path, encoding='utf-8') as file:
            reader = csv.reader(file)

            # read first row to get attributes
            attributes = next(reader)

            # read the data and add entities
            index = 0
            for row in reader:
                entity = Entity(index)
                for i in range(0, len(attributes)):
                    entity.attributes[attributes[i]] = row[i]
                entities.append(entity)
                index += 1

        return entities
