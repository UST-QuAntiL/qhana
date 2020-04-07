from typing import Any
from typing import List
from backend.attribute import Attribute
from backend.logger import Logger
from backend.database import Database
from backend.taxonomie import Taxonomie

"""
This class represents a entity such as a costume or a basic element.
"""
class Entity:
    """
    Initializes the entity object.
    """
    def __init__(self, name: str, id: Any) -> None:
        self.name = name
        self.id = id
        # format: (attribute, None)
        self.attributes = {}
        # format: (attribute, value)
        self.values = {}
        # format: (attribute, base)
        self.bases = {}
        return

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
    Adds abase to the bases list. If the associated
    attribute is not added yet, it will be added.
    """
    def add_base(self, attribute: Attribute, base: Any) -> None:
        if attribute not in self.attributes:
            self.add_attribute(attribute)
        self.bases[attribute] = base
        return

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
    Removes an attribute from the attributes list.
    """
    def remove_base(self, attribute: Attribute) -> None:
        del self.bases[attribute]
        return

    """
    Returns the entity in a single string.
    """
    def __str__(self) -> str:
        output = "name: " + self.name + ", id:" + self.id + ", "
        for attribute_key in self.values:
            output += Attribute.get_name(attribute_key) + ": " + str(self.values[attribute_key]) + ", "
        return output[:-2]

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
TODO: Create this class
"""
class BasicElement(Entity):
    """
    TODO: Create this function
    """
    def __init__(self) -> None:
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
