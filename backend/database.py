from mysql.connector import MySQLConnection
from mysql.connector import Error
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from configparser import ConfigParser
from .costume import Costume
import networkx as nx
from networkx import Graph
from .singleton import Singleton
from .logger import Logger, LogLevel
import datetime

"""
Respresents the database class for db connection
"""
class Database(Singleton):
    def __init__(self) -> None:
        self.connection = None
        self.connected = False
        return
    
    def __del__(self) -> None:
        self.close()
        return

    # Opens the database using the config.ini file
    def open(self, filename = "config.ini") -> None:
        try:
            section = "mysql"
            parser = ConfigParser()
            parser.read(filename)
            connection_string = {}

            if parser.has_section(section):
                items = parser.items(section)
                for item in items:
                    connection_string[item[0]] = item[1]
            else:
                Logger.error('{0} not found in the {1} file'.format(section, filename))
                raise Exception('{0} not found in the {1} file'.format(section, filename))
        
            self.connection = MySQLConnection(**connection_string)
        
            if self.connection.is_connected():
                Logger.debug("Successfully connected to database")
                self.connected = True

        except Error as error:
            Logger.error(str(error) + " The application will quit now.")
            quit()

    def close(self) -> None:
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            self.connected = False
            Logger.debug("Successfully disconnected from database")

    # Returns the table of a taxonomie.
    # In the kostuemrepo a taxonomie table is a "domaene" table with
    # the structure "AlterseindruckName", "UebergeordnetesElement"
    def get_taxonomie_table(self, name: str) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM " + name)
        # Format is (child, parent)
        rows = cursor.fetchall()
        cursor.close()

        return rows

    # Returns the taxonomie, i.e. the root node
    # given the name of the taxonomie table
    def get_graph(self, name: str) -> Graph:
        nodes = dict()
        # Format is (child, parent)
        rows = self.get_taxonomie_table(name)
        graph = nx.DiGraph()
        root_node = None

        for row in rows:
            child = row[0]
            parent = row[1]

            # If parent is none, this is the root node
            if parent is None:
                root_node = child
                nodes[child] = root_node
                rows.remove(row)
                continue

            child_node = nodes.get(child, None)
            if child_node is None:
                child_node = child
                nodes[child] = child_node

            parent_node = nodes.get(parent, None)
            if parent_node is None:
                parent_node = parent
                nodes[parent] = parent_node

            graph.add_node(parent)
            graph.add_node(child)
            graph.add_edge(parent, child)
        
        if root_node is None:
            raise Exception("No root node found in color taxonomie")

        if nx.algorithms.tree.recognition.is_tree(graph) == False:
            raise Exception(name + " is not a tree")
        
        return graph

    def get_color(self) -> Graph:
        return self.get_graph("farbendomaene")
    
    def get_traits(self) -> Graph:
        return self.get_graph("charaktereigenschaftsdomaene")

    def get_condition(self) -> Graph:
        return self.get_graph("zustandsdomaene")

    def get_stereotype(self) -> Graph:
        return self.get_graph("stereotypdomaene")

    def get_gender(self) -> Graph:
        # Creates a graph of gender as there
        # is no taxonomie in the database
        graph = nx.DiGraph()
        graph.add_node("Geschlecht")
        graph.add_node("weiblich")
        graph.add_node("weiblich")
        graph.add_edge("Geschlecht", "weiblich")
        graph.add_edge("Geschlecht", "männlich")
        return graph

    def get_age_impression(self) -> Graph:
        return self.get_graph("alterseindruckdomaene")

    def get_genre(self) -> Graph:
        return self.get_graph("genredomaene")

    def get_job(self) -> Graph:
        return self.get_graph("rollenberufdomaene")

    # Returns a list of all costumes in the database
    # NOTE: Currently there are clones of costumes but I dont know why
    # NOTE: Mybe b.c. there are several costume entries in table Kostuem
    # NOTE: that have the same ID (but why?)
    def get_costumes(self) -> List[Costume]:
        # NOTE: This apprach does not perform good!
        # NOTE: Here, we do 4 transactions to the db for
        # NOTE: each costume entry!
        # TODO: Use different approach to load the costume attributes,
        # TODO: i.e. safe a keys list [KostuemID, RollenID, FilmID]
        # TODO: and iteraite throught this list afterwards.
        costumes : List[Costume] = []
        invalid_entries = 0

        cursor = self.connection.cursor()

        # Get dominant_color and dominant_condition directly from Kostuem table
        query_costume = "SELECT KostuemID, RollenID, FilmID, DominanteFarbe, DominanterZustand FROM Kostuem"
        cursor.execute(query_costume)
        rows_costume = cursor.fetchall()

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
                Logger.warning("Found entry with DominanteFarbe = None. This entry will be skipped.")
                continue
            if row_costume[4] == None:
                invalid_entries += 1
                Logger.warning("Found entry with DominanterZustand = None. This entry will be skipped.")
                continue

            costume = Costume()

            costume.dominant_color = row_costume[3]
            costume.dominant_condition = row_costume[4]

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
                costume.dominant_traits.append(row_trait[0])

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
                costume.stereotypes.append(row_stereotype[0])

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
                
                costume.gender = row_gender_age[0].pop()
                costume.dominant_age_impression = row_gender_age[1]

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
                costume.genres.append(row_genre[0])

            costumes.append(costume)

        cursor.close()

        if (invalid_entries > 0):
            Logger.warning(
                str(invalid_entries) + " costumes from " + str(len(rows_costume)) + " are invalid. " \
                + "These invalid costumes won't be used for further application." \
                )

        return costumes

    # Check if the given costume contains data
    # that cannot be found within the taxonomies
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
