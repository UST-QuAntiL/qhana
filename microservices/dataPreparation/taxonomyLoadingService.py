"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from taxonomy import Taxonomy
from networkx.readwrite import json_graph
import simplejson as json


# we store the mapping between attribute names
# and their taxonomy names with the format
# [attribute_name, taxonomy_name]
attribute_taxonomy_mapping = {
    'Location': 'Ortsbegebenheit',
    'DominantColor': 'Farbe',
    'StereotypeRelevant': 'StereotypRelevant',
    'DominantFunction': 'Funktion',
    'DominantCondition': 'Zustand',
    'DominantCharacterTrait': 'Typus',
    'Stereotyp': 'Stereotyp',
    'Gender': 'Geschlecht',
    'DominantAgeImpression': 'Alterseindruck',
    'Genre': 'Genre',
    'Profession': 'Rollenberuf',
    'RoleRelevance': 'Rollenrelevanz',
    'TimeOfSetting': 'Spielzeit',
    'TimeOfDay': 'Tageszeit',
    'BodyModification': 'Koerpermodifikation',
    'MaritalStatus': 'Familienstand',
    'CharacterTrait': 'Charaktereigenschaft',
    'Venue': 'Spielort',
    'VenueDetail': 'SpielortDetail',
    'AgeImpression': 'Alterseindruck',
    'BaseElement': 'Basiselement',
    'Design': 'Design',
    'Form': 'Form',
    'WayOfWearing': 'Trageweise',
    'Condition': 'Zustand',
    'Function': 'Funktion',
    'Material': 'Material',
    'MaterialImpression': 'Materialeindruck',
    'Color': 'Farbe',
    'ColorImpression': 'Farbeindruck',
    'ColorConcept': 'Farbkonzept'
}


class TaxonomyLoadingService:
    """
    A service class loading taxonomies for the muse repository.
    Moreover, this service encapsulates the logic which attribute
    is mapped to which taxonomy.
    """

    def __init__(self):
        # we store a taxonomy pool for caching purposes.
        # we use the format [name, object].
        self.__taxonomy_pool = dict()

    def _load_from_json(self, name, directory: str = "taxonomies"):
        """
        Loads a taxonomy object from the given json file and name.
        If no directory is specified, the default is ./taxonomies/{name}.json.
        If the taxonomy is already in the pool, it will be used from there instead.
        """

        if str(name) in self.__taxonomy_pool:
            return self.__taxonomy_pool[str(name)]

        name_with_extension = str(name) + ".json"
        file_name = directory + "/" + name_with_extension

        with open(file_name) as file_object:
            graph_json = json.load(file_object)

        graph = json_graph.node_link_graph(graph_json)

        new_taxonomy = Taxonomy(graph)
        self.__taxonomy_pool[str(name)] = new_taxonomy

        return new_taxonomy

    def load(self, attribute):
        """
        Loads the taxonomy that correspond to the given attribute name.
        """

        return self._load_from_json(attribute_taxonomy_mapping[str(attribute)])