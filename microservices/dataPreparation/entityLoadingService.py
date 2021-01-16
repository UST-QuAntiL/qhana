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
    def load_entities(cls, csv_file_name):
        """
        Load all entities from the given csv file.
        We always load all attributes we find in the file.
        We return a list of entity objects.
        """

        entities = []
        with open(csv_file_name, newline=' ') as file:
            reader = csv.reader(file, delimiter=' ', quotechar='|')

            # read first row to get attributes
            header = next(reader)
            header = header.replace(' ', '')
            attributes = header.split(',')

            # read the data and add entities
            index = 0
            for row in reader:
                entity = Entity(str(index), str(index))
                for attribute in attributes:
                    entity.attributes[attribute] = row[attribute]
                entities.append(entity)
                index += 1

        return entities
