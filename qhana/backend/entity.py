from typing import Any, Dict
from typing import List
from qhana.backend.attribute import Attribute
from qhana.backend.logger import Logger
from qhana.backend.database import Database
from qhana.backend.taxonomie import Taxonomie
from datetime import datetime, timedelta
from qhana.backend.attribute import Attribute
import copy

MUSE_URL = "http://129.69.214.108/"


"""
This class represents a entity such as a costume or a basic element.
"""
class Entity:
    """
    Initializes the entity object.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.id = 0
        self.kostuemId = 0
        self.rollenId = 0
        self.filmId = 0
        # format: (attribute, None)
        self.attributes = {}
        # format: (attribute, value)
        self.values = {}

        return

    """
    Sets the id of the entity.
    """
    def set_id(self, id: int) -> None:
        self.id = id
        return

    """
    Gets the id of the entity.
    """
    def get_id(self) -> int:
        return self.id

    """
    Sets the basiselementId of the entity.
    """
    def set_basiselement_id(self, basiselementId: int) -> None:
        self.basiselementId = basiselementId
        return

    """
    Gets the basiselementId of the entity.
    """
    def get_basiselement_id(self) -> int:
        return self.basiselementId

    """
    Sets the kostuemId of the entity.
    """
    def set_kostuem_id(self, kostuemId: int) -> None:
        self.kostuemId = kostuemId
        return

    """
    Gets the kostuemId of the entity.
    """
    def get_kostuem_id(self) -> int:
        return self.kostuemId

    """
    Sets the rollenId of the entity.
    """
    def set_rollen_id(self, rollenId: int) -> None:
        self.rollenId = rollenId
        return

    """
    Gets the rollenId of the entity.
    """
    def get_rollen_id(self) -> int:
        return self.rollenId

    """
    Sets the filmId of the entity.
    """
    def set_film_id(self, filmId: int) -> None:
        self.filmId = filmId
        return

    """
    Gets the filmid of the entity.
    """
    def get_film_id(self) -> int:
        return self.filmId

    """
    Gets the film url of the entity.
    """
    def get_film_url(self) -> str:
        return MUSE_URL + "#/filme/" + str(self.get_film_id())

    """
    Gets the rollen url of the entity.
    """
    def get_rollen_url(self) -> str:
        return self.get_film_url() + "/rollen/" + str(self.get_rollen_id())

    """
    Gets the kostuem url of the entity.
    """
    def get_kostuem_url(self) -> str:
        return self.get_rollen_url() + "/kostueme/" + str(self.get_kostuem_id())

    """
    Adds an attribute to the attributes list.
    """
    def add_attribute(self, attribute: Attribute) -> None:
        self.attributes[attribute] = None
        return

    """
    Adds a value to the values list. If the associated
    attribute is not added yet, it will be added.
    """
    def add_value(self, attribute: Attribute, value: Any) -> None:
        if attribute not in self.attributes:
            self.add_attribute(attribute)
        self.values[attribute] = value
        return

    """
    Gets the value of the given attribute.
    """
    def get_value(self, attribute: Attribute):
        return self.values[attribute]

    """
    Removes an attribute from the attributes list.
    """
    def remove_attribute(self, attribute: Attribute) -> None:
        del self.attributes[attribute]
        return

    """
    Removes a value from the values list.
    """
    def remove_attribute(self, attribute: Attribute) -> None:
        del self.values[attribute]
        return

    """
    Returns the entity in a single string.
    """
    def __str__(self) -> str:
        output = "name: " + self.name + ", id: " + str(self.id) + ", "
        output += " kostuemId: " + str(self.kostuemId) + ", "
        output += " rollenId: " + str(self.rollenId) + ", "
        output += " filmId: " + str(self.filmId) + ", "
        for attribute_key in self.values:
            output += Attribute.get_name(attribute_key) + ": " + str(self.values[attribute_key]) + ", "
        return output[:-2]

"""
This class creats entities based on the given attributes.
"""
class EntityFactory:
    """
    Creates a entity based on the given list of attributes.
    The default for amount is max int, all entities that are found will be returned.
    A filter for the keys can be specified which is a set with the format
    { (KostuemID, RollenID, FilmID) , ... }
    """

    @staticmethod
    def _is_value_in_list_of_values(v: str, lv: List[str]) -> bool:
        for v2 in lv:
            if v.lower() == v2.lower():
                return True

        return False

    @staticmethod
    def _expand_filter_term(attribute: Attribute, filter_term: str, db: Database) -> List[str]:
        if filter_term.startswith("*"):
            value = filter_term[1:].lower()
            taxonomy_type = Attribute.get_taxonomie_type(attribute)
            taxonomy = Taxonomie.create_from_db(taxonomy_type, db)

            lowercase_adj = {}

            k: str
            v: Dict

            for k, v in taxonomy.graph.adj.items():
                lowercase_adj[k.lower()] = [item.lower() for item in v.keys()]

            processed_nodes = []
            new_nodes = [value]

            while len(new_nodes) > 0:
                current_node = new_nodes.pop()

                new_nodes.extend(lowercase_adj[current_node])
                processed_nodes.append(current_node)

            return processed_nodes
        else:
            return [filter_term]

    @staticmethod
    def _is_accepted_by_filter(entity: Entity, filter_rules: Dict[Attribute, List[str]], db: Database) -> bool:
        """
        Tests if the entity is accepted by the provided filter rules or if it should be filtered out.
        :param entity: entity
        :param filter_rules: filter rules
        :return: True if accepted, false if not.
        """
        for attribute in filter_rules:
            attr_accepted = False
            expanded_rules = [
                new_term
                for term in filter_rules[attribute]
                for new_term in EntityFactory._expand_filter_term(attribute, term, db)]

            for value in expanded_rules:
                if EntityFactory._is_value_in_list_of_values(value, entity.get_value(attribute)) or value == "":
                    attr_accepted = True

            if not attr_accepted:
                return False

        return True

    @staticmethod
    def create(
        attributes: List[Attribute],
        database: Database,
        amount: int = 2147483646,
        keys=None,
        filter_rules: Dict[Attribute, List[str]] = {}
        ) -> List[Entity]:

        entities = []
        invalid_entries = 0
        cursor = database.get_cursor()

        # Get KostuemTable
        query_costume = "SELECT " \
            + "KostuemID, RollenID, FilmID, " \
            + "Ortsbegebenheit, DominanteFarbe, " \
            + "StereotypRelevant, DominanteFunktion, " \
            + "DominanterZustand FROM Kostuem"

        if keys is not None:
            keysString = "("
            for i in range(0,len(keys)):
                keysString = keysString + "(" \
                    + str(keys[i][0]) + "," \
                    + str(keys[i][1]) + "," \
                    + str(keys[i][2]) + "),"
            
            keysString = keysString[:-1] + ")"

            query_costume = query_costume \
                + " WHERE (KostuemID, RollenID, FilmID) in " \
                + keysString

        cursor.execute(query_costume)
        rows_costume = cursor.fetchall()

        # i.e. print 10 steps of loading
        printMod = 10

        count = 0

        # First, run as we are just collecting
        # costumes. If there is an attribute set
        # that is just part of a basiselement,
        # this flag will be changed and all
        # entities will be trated as basiselements.
        be_basiselement = False

        # flag that indicates if we break after
        # reaching the required amount of
        # basiselements insteadof 
        # using costumes
        finished = False

        for row_costume in rows_costume:
            if row_costume[0] == None:
                invalid_entries += 1
                Logger.warning("Found entry with KostuemID = None. This entry will be skipped.")
                continue
            if row_costume[1] == None:
                invalid_entries += 1
                Logger.warning("Found entry with RollenID = None. This entry will be skipped.")
                continue
            if row_costume[2] == None:
                invalid_entries += 1
                Logger.warning("Found entry with FilmID = None. This entry will be skipped.")
                continue
            if row_costume[3] == None:
                invalid_entries += 1
                Logger.warning("Found entry with Ortsbegebenheit = None.")
            if row_costume[4] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanteFarbe = None.")
            if row_costume[5] == None:
                invalid_entries += 1
                Logger.warning("Found entry with StereotypRelevant = None.")
            if row_costume[6] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanteFunktion = None.")
            if row_costume[7] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanterZustand = None.")

            kostuemId = row_costume[0]
            rollenId = row_costume[1]
            filmId = row_costume[2]
            ortsbegebenheit = row_costume[3]
            dominanteFarbe = row_costume[4]
            stereotypRelevant = row_costume[5]
            dominanteFunktion = row_costume[6]
            dominanterZustand = row_costume[7]

            entity = Entity("Entity")
            entity.set_film_id(filmId)
            entity.set_rollen_id(rollenId)
            entity.set_kostuem_id(kostuemId)

            if Attribute.ortsbegebenheit in attributes:
                entity.add_attribute(Attribute.ortsbegebenheit)
                if ortsbegebenheit is None:
                    entity.add_value(Attribute.ortsbegebenheit, [])
                else:
                    entity.add_value(Attribute.ortsbegebenheit, list(ortsbegebenheit))
            if Attribute.dominanteFarbe in attributes:
                entity.add_attribute(Attribute.dominanteFarbe)
                if dominanteFarbe is None:
                    entity.add_value(Attribute.dominanteFarbe, [])
                else:
                    entity.add_value(Attribute.dominanteFarbe, [dominanteFarbe])
            if Attribute.stereotypRelevant in attributes:
                entity.add_attribute(Attribute.stereotypRelevant)
                if stereotypRelevant is None:
                    entity.add_value(Attribute.stereotypRelevant, [])
                else:
                    entity.add_value(Attribute.stereotypRelevant, list(stereotypRelevant))
            if Attribute.dominanteFunktion in attributes:
                entity.add_attribute(Attribute.dominanteFunktion)
                if dominanteFunktion is None:
                    entity.add_value(Attribute.dominanteFunktion, [])
                else:
                    entity.add_value(Attribute.dominanteFunktion, [dominanteFunktion])
            if Attribute.dominanterZustand in attributes:
                entity.add_attribute(Attribute.dominanterZustand)
                if dominanterZustand is None:
                    entity.add_value(Attribute.dominanterZustand, [])
                else:
                    entity.add_value(Attribute.dominanterZustand, [dominanterZustand])

            # load farbkonzept if needed
            if Attribute.farbkonzept in attributes:
                query = "SELECT Farbkonzept FROM FilmFarbkonzept "
                query += "WHERE FilmID = %s"
                cursor.execute(query % row_costume[2])
                rows = cursor.fetchall()

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no Farbkonzept. Associated entries are: " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
            
                farbkonzepte = set()

                for row in rows:
                    farbkonzepte.add(row[0])

                entity.add_attribute(Attribute.farbkonzept)
                entity.add_value(Attribute.farbkonzept, list(farbkonzepte))
 
            # load dominanteCharaktereigenschaft if needed
            if Attribute.dominanteCharaktereigenschaft in attributes:
                query_trait = "SELECT DominanteCharaktereigenschaft FROM RolleDominanteCharaktereigenschaft "
                query_trait += "WHERE RollenID = %s AND FilmID = %s"
                cursor.execute(query_trait, (row_costume[1], row_costume[2]))
                rows_trait = cursor.fetchall()

                if len(rows_trait) == 0:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no DominanteCharaktereigenschaft. Associated entries are: " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
            
                dominanteCharaktereigenschaft = []

                for row_trait in rows_trait:
                    dominanteCharaktereigenschaft.append(row_trait[0])

                entity.add_attribute(Attribute.dominanteCharaktereigenschaft)
                entity.add_value(Attribute.dominanteCharaktereigenschaft, dominanteCharaktereigenschaft)

            # load stereotypes if needed
            if Attribute.stereotyp in attributes:
                query_stereotype = "SELECT Stereotyp FROM RolleStereotyp WHERE RollenID = %s AND FilmID = %s"
                cursor.execute(query_stereotype, (row_costume[1], row_costume[2]))
                rows_stereotype = cursor.fetchall()

                if len(rows_stereotype) == 0:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no Stereotyp. Associated entries are: " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
            
                stereotyp = []

                for row_stereotype in rows_stereotype:
                    stereotyp.append(row_stereotype[0])

                entity.add_attribute(Attribute.stereotyp)
                entity.add_value(Attribute.stereotyp, stereotyp)

            # load rollenberuf, geschlecht, dominanterAlterseindruck 
            # or dominantesAlter if needed
            if  Attribute.rollenberuf in attributes or \
                Attribute.geschlecht in attributes or \
                Attribute.dominanterAlterseindruck in attributes or \
                Attribute.dominantesAlter in attributes:

                    query_gender_age = "SELECT " \
                        + "Rollenberuf, Geschlecht, " \
                        + "DominanterAlterseindruck, DominantesAlter," \
                        + "Rollenrelevanz FROM Rolle WHERE " \
                        + "RollenID = %s AND FilmID = %s"
                    cursor.execute(query_gender_age, (row_costume[1], row_costume[2]))
                    rows_gender_age = cursor.fetchall()

                    if len(rows_gender_age) == 0:
                        invalid_entries += 1
                        Logger.warning(
                            "Found entry with no Geschlecht, DominanterAlterseindruck " \
                            + "DominantesAlter and Rollenrelevanz. " \
                            + "Associated entries are: " \
                            + "RollenID = " + str(row_costume[1]) + ", " \
                            + "FilmID = " + str(row_costume[2]) + ". " \
                        )

                    for row_gender_age in rows_gender_age:
                        if Attribute.rollenberuf in attributes:
                            if row_gender_age[0] == None:
                                invalid_entries += 1
                                Logger.warning(
                                    "Found entry with no Geschlecht. Associated entries are: " \
                                    + "RollenID = " + str(row_costume[1]) + ", " \
                                    + "FilmID = " + str(row_costume[2]) + ". " \
                                )
                        if Attribute.geschlecht in attributes:
                            if row_gender_age[1] == None:
                                invalid_entries += 1
                                Logger.warning(
                                    "Found entry with no DominanterAlterseindruck. Associated entries are: " \
                                    + "RollenID = " + str(row_costume[1]) + ", " \
                                    + "FilmID = " + str(row_costume[2]) + ". " \
                                )
                        if Attribute.dominanterAlterseindruck in attributes:
                            if row_gender_age[2] == None:
                                invalid_entries += 1
                                Logger.warning(
                                    "Found entry with no DominantesAlter. Associated entries are: " \
                                    + "RollenID = " + str(row_costume[1]) + ", " \
                                    + "FilmID = " + str(row_costume[2]) + ". " \
                                )
                        if Attribute.dominantesAlter in attributes:
                            if row_gender_age[3] == None:
                                invalid_entries += 1
                                Logger.warning(
                                    "Found entry with no Rollenrelevanz. Associated entries are: " \
                                    + "RollenID = " + str(row_costume[1]) + ", " \
                                    + "FilmID = " + str(row_costume[2]) + ". " \
                                )
                        
                    rollenberuf = row_gender_age[0]
                    geschlecht = row_gender_age[1]
                    dominanterAlterseindruck = row_gender_age[2]
                    dominantesAlter = row_gender_age[3]
                    rollenrelevanz = row_gender_age[4]

                    if Attribute.rollenberuf in attributes:
                        entity.add_attribute(Attribute.rollenberuf)
                        if rollenberuf is None:
                            entity.add_value(Attribute.rollenberuf, [])
                        else:
                            entity.add_value(Attribute.rollenberuf, [rollenberuf])
                    if Attribute.geschlecht in attributes:
                        entity.add_attribute(Attribute.geschlecht)
                        if geschlecht is None:
                            entity.add_value(Attribute.geschlecht, [])
                        else:
                            entity.add_value(Attribute.geschlecht, list(geschlecht))
                    if Attribute.dominanterAlterseindruck in attributes:
                        entity.add_attribute(Attribute.dominanterAlterseindruck)
                        if dominanterAlterseindruck is None:
                            entity.add_value(Attribute.dominanterAlterseindruck, [])
                        else:
                            entity.add_value(Attribute.dominanterAlterseindruck, [dominanterAlterseindruck])
                    if Attribute.dominantesAlter in attributes:
                        entity.add_attribute(Attribute.dominantesAlter)
                        if dominantesAlter is None:
                            entity.add_value(Attribute.dominantesAlter, [])
                        else:
                            entity.add_value(Attribute.dominantesAlter, [dominantesAlter])
                    if Attribute.rollenrelevanz in attributes:
                        entity.add_attribute(Attribute.rollenrelevanz)
                        if rollenrelevanz is None:
                            entity.add_value(Attribute.rollenrelevanz, [])
                        else:
                            entity.add_value(Attribute.rollenrelevanz, list(rollenrelevanz))

            # load genre if needed
            if Attribute.genre in attributes:
                query_genre = "SELECT Genre FROM FilmGenre WHERE FilmID = %s"
                cursor.execute(query_genre, (row_costume[2], ))
                rows_genre = cursor.fetchall()

                if len(rows_genre) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Genre. Associated entry is: " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )

                genre = []
            
                for row_genre in rows_genre:
                    genre.append(row_genre[0])

                entity.add_attribute(Attribute.genre)
                entity.add_value(Attribute.genre, genre)

            # load spielzeit if needed
            if Attribute.spielzeit in attributes:
                query_spielzeit = "SELECT Spielzeit FROM KostuemSpielzeit " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query_spielzeit, (row_costume[0], row_costume[1], row_costume[2]))
                rows_spielzeit = cursor.fetchall()

                entity.add_attribute(Attribute.spielzeit)

                if len(rows_spielzeit) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no KostuemSpielzeit. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.spielzeit, [])
                else:
                    spielzeit = rows_spielzeit[0][0]
                    entity.add_value(Attribute.spielzeit, [spielzeit])

            # load tageszeit if needed
            if Attribute.tageszeit in attributes:
                query_tageszeit = "SELECT Tageszeit FROM KostuemTageszeit " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query_tageszeit, (row_costume[0], row_costume[1], row_costume[2]))
                rows_tageszeit = cursor.fetchall()

                entity.add_attribute(Attribute.tageszeit)

                if len(rows_tageszeit) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no KostuemTageszeit. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.tageszeit, [])
                else:
                    tageszeit = rows_tageszeit[0][0]
                    entity.add_value(Attribute.tageszeit, [tageszeit])

            # load koerpermodifikation if needed
            if Attribute.koerpermodifikation in attributes:
                query = "SELECT Koerpermodifikationname FROM Kostuemkoerpermodifikation " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[0], row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                entity.add_attribute(Attribute.koerpermodifikation)

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Koerpermodifikationname. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.koerpermodifikation, [])
                else:
                    koerpermodifikation = rows[0][0]
                    entity.add_value(Attribute.koerpermodifikation, [koerpermodifikation])

            # load kostuemZeit if needed
            if Attribute.kostuemZeit in attributes:
                query = "SELECT Timecodeanfang, Timecodeende FROM Kostuemtimecode " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[0], row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                entity.add_attribute(Attribute.kostuemZeit)

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Timecodeanfang and Timecodeende. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.kostuemZeit, [])
                else:
                    # in seconds
                    kostuemZeit = 0

                    for row in rows:
                        timecodeende = row[1]
                        timecodeanfang = row[0]

                        kostuemZeit += int((timecodeende - timecodeanfang).total_seconds())

                    entity.add_value(Attribute.kostuemZeit, [kostuemZeit])

            # load familienstand if needed
            if Attribute.familienstand in attributes:
                query = "SELECT Familienstand FROM RolleFamilienstand " \
                    + "WHERE RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                entity.add_attribute(Attribute.familienstand)

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Familienstand. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.familienstand, [])
                else:
                    familienstaende = []
                    for familienstand in rows[0][0]:
                        familienstaende.append(familienstand)
                    entity.add_value(Attribute.familienstand, familienstaende)

            # load charakterEigenschaft if needed
            if Attribute.charaktereigenschaft in attributes:
                query = "SELECT Charaktereigenschaft FROM KostuemCharaktereigenschaft " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[0], row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                entity.add_attribute(Attribute.charaktereigenschaft)

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Charaktereigenschaft. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    entity.add_value(Attribute.charaktereigenschaft, [])
                else:
                    charaktereigenschaften = []
                    for charaktereigenschaft in rows:
                        charaktereigenschaften.append(charaktereigenschaft[0])
                    entity.add_value(Attribute.charaktereigenschaft, charaktereigenschaften)

            # load spielort or spielortDetail if needed
            if  Attribute.spielort in attributes or \
                Attribute.spielortDetail in attributes:

                query = "SELECT Spielort, SpielortDetail FROM KostuemSpielort " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[0], row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Spielort or SpielortDetail. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    if Attribute.spielort in attributes:
                        entity.add_attribute(Attribute.spielort)
                        entity.add_value(Attribute.spielort, [])
                    
                    if Attribute.spielortDetail in attributes:
                        entity.add_attribute(Attribute.spielortDetail)
                        entity.add_value(Attribute.spielortDetail, [])
                else:
                    # use set to avoid duplicates
                    spielorte = set()
                    spielortdetails = set()

                    for row in rows:
                        spielorte.add(row[0])
                        spielortdetails.add(row[1])

                    if Attribute.spielort in attributes:
                        entity.add_attribute(Attribute.spielort)
                        entity.add_value(Attribute.spielort, list(spielorte))

                    if Attribute.spielortDetail in attributes:
                        entity.add_attribute(Attribute.spielortDetail)
                        entity.add_value(Attribute.spielortDetail, list(spielortdetails))

            # load alterseindruck or alter if needed
            if  Attribute.alterseindruck in attributes or \
                Attribute.alter in attributes:

                query = "SELECT Alterseindruck, NumAlter FROM KostuemAlterseindruck " \
                    + "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query, (row_costume[0], row_costume[1], row_costume[2]))
                rows = cursor.fetchall()

                if len(rows) == 0:
                    invalid_entries += 1
                    Logger.warning( \
                        "Found entry with no Spielort or SpielortDetail. Associated entry is: " \
                        + "KostuemID = " + str(row_costume[0]) + ", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )
                    if Attribute.alterseindruck in attributes:
                        entity.add_attribute(Attribute.alterseindruck)
                        entity.add_value(Attribute.alterseindruck, [])
                    
                    if Attribute.alter in attributes:
                        entity.add_attribute(Attribute.alter)
                        entity.add_value(Attribute.alter, [])
                else:
                    # use set to avoid duplicates
                    alterseindrucke = set()
                    alters = set()

                    for row in rows:
                        alterseindrucke.add(row[0])
                        alters.add(row[1])

                    if Attribute.alterseindruck in attributes:
                        entity.add_attribute(Attribute.alterseindruck)
                        entity.add_value(Attribute.alterseindruck, list(alterseindrucke))

                    if Attribute.alter in attributes:
                        entity.add_attribute(Attribute.alter)
                        entity.add_value(Attribute.alter, list(alters))

            # load basiselement if needed
            # this also means that we are now treat
            # each datapoint as a basiselement
            if  Attribute.basiselement in attributes or \
                Attribute.design in attributes or \
                Attribute.form in attributes or \
                Attribute.trageweise in attributes or \
                Attribute.zustand in attributes or \
                Attribute.funktion in attributes or \
                Attribute.material in attributes or \
                Attribute.materialeindruck in attributes or \
                Attribute.farbe in attributes or \
                Attribute.farbeindruck in attributes:

                be_basiselement = True
                query_trait = "SELECT BasiselementID FROM KostuemBasiselement "
                query_trait += "WHERE KostuemID = %s AND RollenID = %s AND FilmID = %s"
                cursor.execute(query_trait, (row_costume[0], row_costume[1], row_costume[2]))
                rows_basiselement = cursor.fetchall()

                if len(rows_basiselement) == 0:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no Basiselement. Associated entries are: " \
                        + "KostuemID = " + str(row_costume[0]) +", " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                    )

                kostuemID = row_costume[0]
                rollenID = row_costume[1]
                filmID = row_costume[2]

                for row_basiselement in rows_basiselement:
                    # copy entity because every basiselement
                    # will now be an own entity
                    entity_basis = copy.deepcopy(entity)
                    
                    basiselementID = row_basiselement[0]

                    entity_basis.set_basiselement_id(basiselementID)

                    # load basiselementName if needed
                    if Attribute.basiselement in attributes: 
                        query_trait = "SELECT Basiselementname FROM Basiselement "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Basiselementname. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.basiselement)
                            entity_basis.add_value(Attribute.basiselement, [])

                        # use set to avoid duplicates
                        basiselementNames = set()

                        for row in rows:
                            basiselementNames.add(row[0])

                        entity_basis.add_attribute(Attribute.basiselement)
                        entity_basis.add_value(Attribute.basiselement, list(basiselementNames))


                    # load design if needed
                    if Attribute.design in attributes:
                        query_trait = "SELECT Designname FROM BasiselementDesign "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Designname. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.design)
                            entity_basis.add_value(Attribute.design, [])

                        # use set to avoid duplicates
                        designs = set()

                        for row in rows:
                            designs.add(row[0])

                        entity_basis.add_attribute(Attribute.design)
                        entity_basis.add_value(Attribute.design, list(designs))

                    # load form if needed
                    if Attribute.form in attributes:
                        query_trait = "SELECT Formname FROM BasiselementForm "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Formname. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.form)
                            entity_basis.add_value(Attribute.form, [])

                        # use set to avoid duplicates
                        forms = set()

                        for row in rows:
                            forms.add(row[0])

                        entity_basis.add_attribute(Attribute.form)
                        entity_basis.add_value(Attribute.form, list(forms))

                    # load trageweise if needed
                    if Attribute.trageweise in attributes:
                        query_trait = "SELECT Trageweisename FROM BasiselementTrageweise "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Trageweisename. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.trageweise)
                            entity_basis.add_value(Attribute.trageweise, [])

                        # use set to avoid duplicates
                        trageweisen = set()

                        for row in rows:
                            trageweisen.add(row[0])

                        entity_basis.add_attribute(Attribute.trageweise)
                        entity_basis.add_value(Attribute.trageweise, list(trageweisen))

                    # load zustand if needed
                    if Attribute.zustand in attributes:
                        query_trait = "SELECT Zustandsname FROM BasiselementZustand "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Zustandsname. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.zustand)
                            entity_basis.add_value(Attribute.zustand, [])

                        # use set to avoid duplicates
                        zustande = set()

                        for row in rows:
                            zustande.add(row[0])

                        entity_basis.add_attribute(Attribute.zustand)
                        entity_basis.add_value(Attribute.zustand, list(zustande))

                    # load funktion if needed
                    if Attribute.funktion in attributes:
                        query_trait = "SELECT Funktionsname FROM BasiselementFunktion "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Funktionsname. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.funktion)
                            entity_basis.add_value(Attribute.funktion, [])

                        # use set to avoid duplicates
                        funktionen = set()

                        for row in rows:
                            funktionen.add(row[0])

                        entity_basis.add_attribute(Attribute.funktion)
                        entity_basis.add_value(Attribute.funktion, list(funktionen))

                    # load materialName and materialEindruck if needed
                    if  Attribute.material in attributes or \
                        Attribute.materialeindruck in attributes:
                        query_trait = "SELECT Materialname, Materialeindruck FROM BasiselementMaterial "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Materialname and Materialeindruck. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.material)
                            entity_basis.add_value(Attribute.material, [])
                            entity_basis.add_attribute(Attribute.materialeindruck)
                            entity_basis.add_value(Attribute.materialeindruck, [])

                        # use set to avoid duplicates
                        materialien = set()
                        materialEindrucke = set()

                        for row in rows:
                            materialien.add(row[0])
                            materialEindrucke.add(row[1])

                        if Attribute.material in attributes:
                            entity_basis.add_attribute(Attribute.material)
                            entity_basis.add_value(Attribute.material, list(materialien))

                        if Attribute.materialeindruck in attributes:
                            entity_basis.add_attribute(Attribute.materialeindruck)
                            entity_basis.add_value(Attribute.materialeindruck, list(materialEindrucke))

                    # load farbName and farbEindruck if needed
                    if  Attribute.farbe in attributes or \
                        Attribute.farbeindruck in attributes:
                        query_trait = "SELECT Farbname, Farbeindruck FROM BasiselementFarbe "
                        query_trait += "WHERE BasiselementID = %s"
                        cursor.execute(query_trait % basiselementID)
                        rows = cursor.fetchall()

                        if len(rows) == 0:
                            invalid_entries += 1
                            Logger.warning(
                                "Found entry with no Farbname and Farbeindruck. Associated entries are: " \
                                + "BasiselementID = " + str(basiselementID) +", " \
                                + "KostuemID = " + str(kostuemID) +", " \
                                + "RollenID = " + str(rollenID) + ", " \
                                + "FilmID = " + str(filmID) + ". " \
                            )
                            entity_basis.add_attribute(Attribute.farbe)
                            entity_basis.add_value(Attribute.farbe, [])
                            entity_basis.add_attribute(Attribute.farbeindruck)
                            entity_basis.add_value(Attribute.farbeindruck, [])

                        # use set to avoid duplicates
                        farben = set()
                        farbeindrucke = set()

                        for row in rows:
                            farben.add(row[0])
                            farbeindrucke.add(row[1])

                        if Attribute.farbe in attributes:
                            entity_basis.add_attribute(Attribute.farbe)
                            entity_basis.add_value(Attribute.farbe, list(farben))

                        if Attribute.farbeindruck in attributes:
                            entity_basis.add_attribute(Attribute.farbeindruck)
                            entity_basis.add_value(Attribute.farbeindruck, list(farbeindrucke))

                    if EntityFactory._is_accepted_by_filter(entity_basis, filter_rules, database):
                        entity_basis.set_id(count)
                        entities.append(entity_basis)
                        count += 1
                        if count % printMod == 0:
                            Logger.normal(str(count) + " / " + str(amount) + " entities loaded")
                        if count >= amount:
                            finished = True
                            break

            if finished:
                break

            if (not be_basiselement) and EntityFactory._is_accepted_by_filter(entity, filter_rules, database):
                entity.set_id(count)
                entities.append(entity)
                count += 1
                if count % printMod == 0:
                    Logger.normal(str(count) + " / " + str(amount) + " entities loaded")     
                if count >= amount:
                    break

        Logger.normal(str(count) + " entities loaded with " + str(invalid_entries) + " being invalid.")

        return entities

"""
This class represents a costume in the simplified model
NOTE: We use stereotyp although it is not defined in the simplified model
"""
class Costume(Entity):
    """
    Initialize a costume object with the predefined attributes from the
    simplified model.
    """
    def __init__(self, id) -> None:
        super().__init__("Kostüm", id)

        # add all the attributes
        self.add_attribute(Attribute.dominanteFarbe)
        self.add_attribute(Attribute.dominanteCharaktereigenschaft)
        self.add_attribute(Attribute.dominanterZustand)
        self.add_attribute(Attribute.stereotyp)
        self.add_attribute(Attribute.geschlecht)
        self.add_attribute(Attribute.dominanterAlterseindruck)
        self.add_attribute(Attribute.genre)

        return

""" 
Represents the factory to create costumes
"""
class CostumeFactory:
    """
    Creates one single costume.
    """
    @staticmethod
    def __create_costume(
        id: Any,
        dominanteFarbe: [str],
        dominanteCharaktereigenschaft: [str],
        dominanterZustand: [str],
        stereotyp: [str],
        geschlecht: [str],
        dominanterAlterseindruck: [str],
        genre: [str],
        database: Database
        ) -> Costume:

        costume = Costume(id)

        # add all the values
        costume.add_value(Attribute.dominanteFarbe, dominanteFarbe)
        costume.add_value(Attribute.dominanteCharaktereigenschaft, dominanteCharaktereigenschaft)
        costume.add_value(Attribute.dominanterZustand, dominanterZustand)
        costume.add_value(Attribute.stereotyp, stereotyp)
        costume.add_value(Attribute.geschlecht, geschlecht)
        costume.add_value(Attribute.dominanterAlterseindruck, dominanterAlterseindruck)
        costume.add_value(Attribute.genre, genre)

        # add all the bases, i.e. taxonomies
        costume.add_base(Attribute.dominanteFarbe, Attribute.get_base(Attribute.dominanteFarbe, database))
        costume.add_base(Attribute.dominanteCharaktereigenschaft, Attribute.get_base(Attribute.dominanteCharaktereigenschaft, database))
        costume.add_base(Attribute.dominanterZustand, Attribute.get_base(Attribute.dominanterZustand, database))
        costume.add_base(Attribute.stereotyp, Attribute.get_base(Attribute.stereotyp, database))
        costume.add_base(Attribute.geschlecht, Attribute.get_base(Attribute.geschlecht, database))
        costume.add_base(Attribute.dominanterAlterseindruck, Attribute.get_base(Attribute.dominanterAlterseindruck, database))
        costume.add_base(Attribute.genre, Attribute.get_base(Attribute.genre, database))

        return costume

    """
    Create all costumes from the database.
    NOTE: Currently there are clones of costumes but I dont know why
    NOTE: Mybe b.c. there are several costume entries in table Kostuem
    NOTE: that have the same ID (but why?)
    TODO: Implement filter possibility
    """
    @staticmethod
    def create(database: Database, amount: int = 0) -> List[Costume]:
        # NOTE: This apprach does not perform good!
        # NOTE: Here, we do 4 transactions to the db for
        # NOTE: each costume entry!
        # TODO: Use different approach to load the costume attributes,
        # TODO: i.e. safe a keys list [KostuemID, RollenID, FilmID]
        # TODO: and iteraite throught this list afterwards.
        costumes: List[Costume] = []
        invalid_entries = 0

        cursor = database.get_cursor()

        # Get dominant_color and dominant_condition directly from Kostuem table
        query_costume = "SELECT KostuemID, RollenID, FilmID, DominanteFarbe, DominanterZustand FROM Kostuem"
        cursor.execute(query_costume)
        rows_costume = cursor.fetchall()

        count = 0
        if amount <= 0:
            amount = 2147483646

        for row_costume in rows_costume:
            if count == amount:
                break
            if row_costume[0] == None:
                invalid_entries += 1
                Logger.warning("Found entry with KostuemID = None. This entry will be skipped.")
                continue
            if row_costume[1] == None:
                invalid_entries += 1
                Logger.warning("Found entry with RollenID = None. This entry will be skipped.")
                continue
            if row_costume[2] == None:
                invalid_entries += 1
                Logger.warning("Found entry with FilmID = None. This entry will be skipped.")
                continue
            if row_costume[3] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanteFarbe = None. This entry will be skipped.")
                continue
            if row_costume[4] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanterZustand = None. This entry will be skipped.")
                continue

            kostuemId = row_costume[0]
            rollenId = row_costume[1]
            filmId = row_costume[2]
            dominanteFarbe = [row_costume[3]]
            dominanterZustand = [row_costume[4]]
            dominanteCharaktereigenschaft = []
            stereotyp = []
            geschlecht = []
            dominanterAlterseindruck = []
            genre = []

            # Get dominant_traits from table RolleDominanteCharaktereigenschaften
            query_trait = "SELECT DominanteCharaktereigenschaft FROM RolleDominanteCharaktereigenschaft "
            query_trait += "WHERE RollenID = %s AND FilmID = %s"
            cursor.execute(query_trait, (row_costume[1], row_costume[2]))
            rows_trait = cursor.fetchall()

            if len(rows_trait) == 0:
                invalid_entries += 1
                Logger.warning(
                    "Found entry with no DominanteCharaktereigenschaft. Associated entries are: " \
                    + "RollenID = " + str(row_costume[1]) + ", " \
                    + "FilmID = " + str(row_costume[2]) + ". " \
                    + "This entry will be skipped." \
                )
                continue
        
            for row_trait in rows_trait:
                dominanteCharaktereigenschaft.append(row_trait[0])

            # Get stereotypes from table RolleStereotyp
            query_stereotype = "SELECT Stereotyp FROM RolleStereotyp WHERE RollenID = %s AND FilmID = %s"
            cursor.execute(query_stereotype, (row_costume[1], row_costume[2]))
            rows_stereotype = cursor.fetchall()

            if len(rows_stereotype) == 0:
                invalid_entries += 1
                Logger.warning(
                    "Found entry with no Stereotyp. Associated entries are: " \
                    + "RollenID = " + str(row_costume[1]) + ", " \
                    + "FilmID = " + str(row_costume[2]) + ". " \
                    + "This entry will be skipped." \
                )
                continue
        
            for row_stereotype in rows_stereotype:
                stereotyp.append(row_stereotype[0])

            # Get gender and dominant_age_impression from table Rolle
            query_gender_age = "SELECT Geschlecht, DominanterAlterseindruck FROM Rolle WHERE "
            query_gender_age += "RollenID = %s AND FilmID = %s"
            cursor.execute(query_gender_age, (row_costume[1], row_costume[2]))
            rows_gender_age = cursor.fetchall()

            if len(rows_gender_age) == 0:
                invalid_entries += 1
                Logger.warning(
                    "Found entry with no Geschlecht and no DominanterAlterseindruck. Associated entries are: " \
                    + "RollenID = " + str(row_costume[1]) + ", " \
                    + "FilmID = " + str(row_costume[2]) + ". " \
                    + "This entry will be skipped." \
                )
                continue
        
            for row_gender_age in rows_gender_age:
                if row_gender_age[0] == None:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no Geschlecht. Associated entries are: " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                        + "This entry will be skipped." \
                    )
                    continue
                if row_gender_age[1] == None:
                    invalid_entries += 1
                    Logger.warning(
                        "Found entry with no DominanterAlterseindruck. Associated entries are: " \
                        + "RollenID = " + str(row_costume[1]) + ", " \
                        + "FilmID = " + str(row_costume[2]) + ". " \
                        + "This entry will be skipped." \
                    )
                    continue
                
                geschlecht.append(row_gender_age[0].pop())
                dominanterAlterseindruck.append(row_gender_age[1])

            # Get genres from table FilmGenre
            query_genre = "SELECT Genre FROM FilmGenre WHERE FilmID = %s"
            cursor.execute(query_genre, (row_costume[2], ))
            rows_genre = cursor.fetchall()

            if len(rows_genre) == 0:
                invalid_entries += 1
                Logger.warning( \
                    "Found entry with no Genre. Associated entry is: " \
                    + "FilmID = " + str(row_costume[2]) + ". " \
                    + "This entry will be skipped." \
                )
                continue
        
            for row_genre in rows_genre:
                genre.append(row_genre[0])

            costume_id = str(kostuemId) + ":" + str(rollenId) + ":" + str(filmId)

            costume = CostumeFactory.__create_costume(
                costume_id,
                dominanteFarbe,
                dominanteCharaktereigenschaft,
                dominanterZustand,
                stereotyp,
                geschlecht,
                dominanterAlterseindruck,
                genre,
                database
            )

            costumes.append(costume)
            count += 1

        cursor.close()

        if (invalid_entries > 0):
            Logger.warning(
                str(invalid_entries) + " costumes from " + str(count) + " are invalid. " \
                + "These invalid costumes won't be used for further application." \
                )

        return costumes

    # Check if the given costume contains data
    # that cannot be found within the taxonomies
    @staticmethod
    def validate_costum(costume) -> (bool, str):
        # load all taxonomies
        color = self.get_color()
        traits = self.get_traits()
        condition = self.get_condition()
        stereotype = self.get_stereotype()
        gender = self.get_gender()
        age_impression = self.get_age_impression()
        genre = self.get_genre()

        if not color.has_node(costume.dominant_color):
            return (False, "cominant_color")
        
        for trait in costume.dominant_traits:
            if not traits.has_node(trait):
                return (False, "dominant_trait")

        if not condition.has_node(costume.dominant_condition):
            return (False, "dominant_condition")

        for _stereotype in costume.stereotypes:
            if not stereotype.has_node(_stereotype):
                return (False, "stereotype")

        if not gender.has_node(costume.gender):
            return (False, "gender")

        if not age_impression.has_node(costume.dominant_age_impression):
            return (False, "dominant_age_impression")

        for _genre in costume.genres:
            if not genre.has_node(_genre):
                return (False, "genre")
        
        return (True, "")

    # Compares the costumes and their data with the
    # taxonomies and check for inconsistencies
    # file: specifies a separate file for the output
    # if None: log/yyyy_mm_dd_hh_ss_db_validation.log
    # will be used
    @staticmethod
    def validate_all_costumes(self, file=None) -> None:
        Logger.normal("Running validation of the database.")

        if self.connected == False:
            Logger.error("Validation not possible because we are currently not connected to a database.")
        if file == None:
            file = "log/" + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "_db_validation.log"

        # load all taxonomies
        color = self.get_color()
        traits = self.get_traits()
        condition = self.get_condition()
        stereotype = self.get_stereotype()
        gender = self.get_gender()
        age_impression = self.get_age_impression()
        genre = self.get_genre()

        # load all costumes
        costumes = self.get_costumes()

        output_string = ""

        found_errors = []

        invalid_costumes = 0
        is_current_costume_invalid = False

        # compare all costumes with taxonomies
        for costume in costumes:
            if not color.has_node(costume.dominant_color):
                error = costume.dominant_color + " could not be found in color taxonomie. "
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"
            
            for trait in costume.dominant_traits:
                if not traits.has_node(trait):
                    error = trait + " could not be found in traits taxonomie. "
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"

            if not condition.has_node(costume.dominant_condition):
                error = costume.dominant_condition + " could not be found in condition taxonomie. "
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            for _stereotype in costume.stereotypes:
                if not stereotype.has_node(_stereotype):
                    error = _stereotype + " could not be found in stereotype taxonomie. "
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"

            if not gender.has_node(costume.gender):
                error = costume.gender + " could not be found in gender taxonomie. "
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            if not age_impression.has_node(costume.dominant_age_impression):
                error = costume.dominant_age_impression + " could not be found in age_impression taxonomie. "
                is_current_costume_invalid = True
                if error not in found_errors:
                    found_errors.append(error)
                    output_string += error + " costume: " + str(costume) + "\n"

            for _genre in costume.genres:
                if not genre.has_node(_genre):
                    error = _genre + " could not be found in genre taxonomie. "
                    is_current_costume_invalid = True
                    is_current_costume_invalid = True
                    if error not in found_errors:
                        found_errors.append(error)
                        output_string += error + " costume: " + str(costume) + "\n"
            
            if is_current_costume_invalid:
                invalid_costumes += 1
                is_current_costume_invalid = False

        with open(file, "w+") as file_object:
            file_object.write(output_string)

        Logger.normal(str(invalid_costumes) + " / " + str(len(costumes)) + " are invalid")
        Logger.normal("Validation output has been written to " + file)

        return