"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from taxonomy import Taxonomy
from attribute import Attribute
from networkx.readwrite import json_graph
import simplejson as json


# we store the mapping between attributes and
# their taxonomy names with the format
# [Attribute, taxonomy_name]
attribute_taxonomy_mapping = {
    Attribute.Location: 'Ortsbegebenheit',
    Attribute.DominantColor: 'Farbe',
    Attribute.StereotypeRelevant: 'StereotypRelevant',
    Attribute.DominantFunction: 'Funktion',
    Attribute.DominantCondition: 'Zustand',
    Attribute.DominantCharacterTrait: 'Typus',
    Attribute.Stereotyp: 'Stereotyp',
    Attribute.Gender: 'Geschlecht',
    Attribute.DominantAgeImpression: 'Alterseindruck',
    Attribute.Genre: 'Genre',
    Attribute.Profession: 'Rollenberuf',
    Attribute.RoleRelevance: 'Rollenrelevanz',
    Attribute.TimeOfSetting: 'Spielzeit',
    Attribute.TimeOfDay: 'Tageszeit',
    Attribute.BodyModification: 'Koerpermodifikation',
    Attribute.MaritalStatus: 'Familienstand',
    Attribute.CharacterTrait: 'Charaktereigenschaft',
    Attribute.Venue: 'Spielort',
    Attribute.VenueDetail: 'SpielortDetail',
    Attribute.AgeImpression: 'Alterseindruck',
    Attribute.BaseElement: 'Basiselement',
    Attribute.Design: 'Design',
    Attribute.Form: 'Form',
    Attribute.WayOfWearing: 'Trageweise',
    Attribute.Condition: 'Zustand',
    Attribute.Function: 'Funktion',
    Attribute.Material: 'Material',
    Attribute.MaterialImpression: 'Materialeindruck',
    Attribute.Color: 'Farbe',
    Attribute.ColorImpression: 'Farbeindruck',
    Attribute.ColorConcept: 'Farbkonzept'
}


class TaxonomyService:
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

        return self._load_from_json(attribute_taxonomy_mapping[attribute])