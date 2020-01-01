from mysql.connector import MySQLConnection
from mysql.connector import Error
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from configparser import ConfigParser
from taxonomie import Taxonomie
from costume import Costume

class Database:
    def __init__(self) -> None:
        self.connection = None
    
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
                raise Exception('{0} not found in the {1} file'.format(section, filename))
        
            self.connection = MySQLConnection(**connection_string)
        
            if self.connection.is_connected():
                print("Successfully connected to database")

        except Error as error:
            print(error)
            quit()

    def close(self) -> None:
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            print("Successfully disconnected from database ")

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
    def get_taxonomie(self, name: str) -> Taxonomie:
        nodes = dict()
        # Format is (child, parent)
        rows = self.get_taxonomie_table(name)
        root_node = None

        for row in rows:
            child = row[0]
            parent = row[1]

            # If parent is none, this is the root node
            if parent is None:
                root_node = Taxonomie(child)
                nodes[child] = root_node
                rows.remove(row)
                continue

            child_node = nodes.get(child, None)
            if child_node is None:
                child_node = Taxonomie(child)
                nodes[child] = child_node

            parent_node = nodes.get(parent, None)
            if parent_node is None:
                parent_node = Taxonomie(parent)
                nodes[parent] = parent_node

            parent_node.add_child(child_node)
        
        if root_node is None:
            raise Exception("No root node found in color taxonomie")
        
        return root_node

    def get_color(self) -> Taxonomie:
        return self.get_taxonomie("farbendomaene")
    
    def get_traits(self) -> Taxonomie:
        return self.get_taxonomie("charaktereigenschaftsdomaene")

    def get_condition(self) -> Taxonomie:
        return self.get_taxonomie("zustandsdomaene")

    def get_stereotype(self) -> Taxonomie:
        return self.get_taxonomie("stereotypdomaene")

    def get_gender(self) -> Taxonomie:
        # Creates a taxonomie of gender as there
        # is no taxonomie in the database
        root_node = Taxonomie("Geschlecht")
        root_node.add_child(Taxonomie("weiblich"))
        root_node.add_child(Taxonomie("männlich"))
        return root_node

    def get_age_impression(self) -> Taxonomie:
        return self.get_taxonomie("alterseindruckdomaene")

    def get_genre(self) -> Taxonomie:
        return self.get_taxonomie("genredomaene")

    # Returns a list of all costumes in the database
    # NOTE: Currently there are clones of costumes but i dont know why
    # NOTE: Mybe b.c. there are several costume entries in table Kostuem
    # NOTE: that have the same ID (but why?)
    def get_costumes(self) -> List[Costume]:
        costumes : List[Costume] = []
        invalid_entries = 0

        cursor = self.connection.cursor()

        # Get dominant_color and dominant_condition directly from Kostuem table
        query_costume = "SELECT KostuemID, RollenID, FilmID, DominanteFarbe, DominanterZustand FROM Kostuem"
        cursor.execute(query_costume)
        rows_costume = cursor.fetchall()

        for row_costume in rows_costume:
            if row_costume[0] == None \
            or row_costume[1] == None \
            or row_costume[2] == None \
            or row_costume[3] == None \
            or row_costume[4] == None:
                invalid_entries += 1
                continue
            else:
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
                    continue
            
                for row_trait in rows_trait:
                    costume.dominant_traits.append(row_trait[0])

                # Get stereotypes from table RolleStereotyp
                query_stereotype = "SELECT Stereotyp FROM RolleStereotyp WHERE RollenID = %s AND FilmID = %s"
                cursor.execute(query_stereotype, (row_costume[1], row_costume[2]))
                rows_stereotype = cursor.fetchall()

                if len(rows_stereotype) == 0:
                    invalid_entries += 1
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
                    continue
            
                for row_gender_age in rows_gender_age:
                    if row_gender_age[0] == None \
                    or row_gender_age[1] == None:
                        invalid_entries += 1
                        continue
                    else:
                        costume.gender = row_gender_age[0]
                        costume.dominant_age_impression = row_gender_age[1]

                # Get genres from table FilmGenre
                query_genre = "SELECT Genre FROM FilmGenre WHERE FilmID = %s"
                cursor.execute(query_genre, (row_costume[2], ))
                rows_genre = cursor.fetchall()

                if len(rows_genre) == 0:
                    invalid_entries += 1
                    continue
            
                for row_genre in rows_genre:
                    costume.genres.append(row_genre[0])

                costumes.append(costume)

        cursor.close()

        print(str(invalid_entries) + " from " + str(len(rows_costume)) + " are invalid")

        return costumes

if __name__ == '__main__':
    database = Database()
    database.open()
    costumes = database.get_costumes()

    for costume in costumes:
        print(costume)

    database.close()
