from mysql.connector import MySQLConnection
from mysql.connector import Error
from typing import Tuple
from typing import List
from typing import Dict
from configparser import ConfigParser
from taxonomie import Taxonomie

class Database:
    def __init__(self) -> None:
        self.connection = None
    
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

    def close(self) -> None:
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            print("Successfully disconnected from database ")

    def get_taxonomie_table(self, name: str) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM " + name)
            # Format is (child, parent)
            rows = cursor.fetchall()

        except Error as error:
            print(error)

        return rows

    def get_taxonomie(self, name: str):
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

    def get_color(self):
        return self.get_taxonomie("farbendomaene")
    
    def get_traits(self):
        return self.get_taxonomie("charaktereigenschaftsdomaene")

    def get_condition(self):
        return self.get_taxonomie("zustandsdomaene")

    def get_stereotype(self):
        return self.get_taxonomie("stereotypdomaene")

    def get_gender(self):
        root_node = Taxonomie("Geschlecht")
        root_node.add_child(Taxonomie("weiblich"))
        root_node.add_child(Taxonomie("männlich"))
        return root_node

    def get_age_impression(self):
        return self.get_taxonomie("alterseindruckdomaene")

    def get_genre(self):
        return self.get_taxonomie("genredomaene")

if __name__ == '__main__':
    database = Database()
    database.open()
    color = database.get_gender()
    print(color)
    database.close()
