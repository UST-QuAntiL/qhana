from typing import Any
import networkx as nx
from networkx import Graph
from database import Database
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import subprocess
import os
from networkx.readwrite import json_graph
import simplejson as json
import enum

class Attribute(enum.Enum):
    color = 1
    traits = 2
    condition = 3
    stereotype = 4
    gender = 5
    age_impression = 6
    genre = 7

class Taxonomie():
    def __init__(self) -> None:
        self.database: Database = Database()
        self.plot_datatype: str = "png"
        self.plot_directory: str = "plots"
        self.graph_directory: str = "graphs"
        self.color: Graph = None
        self.traits: Graph = None
        self.condition: Graph = None
        self.stereotype: Graph = None
        self.gender: Graph = None
        self.age_impression: Graph = None
        self.genre: Graph = None
        return

    def __del__(self) -> None:
        del self.database
        return
    
    # Loads the taxonomies from database
    def load_from_database(self) -> None:
        self.database.open()
        self.color = self.database.get_color()
        self.traits = self.database.get_traits()
        self.condition = self.database.get_condition()
        self.stereotype = self.database.get_stereotype()
        self.gender = self.database.get_gender()
        self.age_impression = self.database.get_age_impression()
        self.genre = self.database.get_genre()
        self.database.close()

    # Gets the root of the given graph (tree)
    def get_root(self, graph: Graph) -> str:
        return [n for n,d in graph.in_degree() if d == 0][0]

    # Gets the graph corresponding to the attribute
    def get_graph(self, attribute: Attribute) -> Graph:
        if attribute == Attribute.color:
            return self.color
        elif attribute == Attribute.traits:
            return self.traits
        elif attribute == Attribute.condition:
            return self.condition
        elif attribute == Attribute.stereotype:
            return self.stereotype
        elif attribute == Attribute.gender:
            return self.gender
        elif attribute == Attribute.age_impression:
            return self.age_impression
        elif attribute == Attribute.genre:
            return self.genre
        else:
            raise Exception("Unkown attribute")

    # Sets the graph corresponding to the attribute
    def set_graph(self, attribute: Attribute, graph: Graph) -> None:
        if attribute == Attribute.color:
            self.color = graph
        elif attribute == Attribute.traits:
            self.traits = graph
        elif attribute == Attribute.condition:
            self.condition = graph
        elif attribute == Attribute.stereotype:
            self.stereotype = graph
        elif attribute == Attribute.gender:
            self.gender = graph
        elif attribute == Attribute.age_impression:
            self.age_impression = graph
        elif attribute == Attribute.genre:
            self.genre = graph
        else:
            raise Exception("Unkown attribute")
        return

    # Safe the graph to the given attribute as dot 
    # and png file and displays the created image
    def plot(self, attribute: Attribute, display: bool = True) -> None:
        if os.path.isdir(self.plot_directory) == False:
            os.mkdir(self.plot_directory)

        name = attribute.name

        filepath: str = self.plot_directory + "/" + name

        graph = self.get_graph(attribute)

        nx.nx_agraph.write_dot(graph, filepath + ".dot")
        subprocess.call(["dot", "-T" + self.plot_datatype, filepath + ".dot", "-o", filepath + "." + self.plot_datatype])

        if display == True:
            img = Image.open(filepath + "." + self.plot_datatype)
            img.show()
        
        return
    
    # Writes the graph to a data file
    def safe_to_file(self,  attribute: Attribute) -> None:
        if os.path.isdir(self.graph_directory) == False:
            os.mkdir(self.graph_directory)

        name = attribute.name

        filename: str = self.graph_directory + "/" + name + ".json"

        graph = self.get_graph(attribute)

        graph_json = json_graph.node_link_data(graph)

        json.dump(graph_json, open(filename,'w'), indent = 2)

        return

    # Reads the graph from a data file
    def load_from_file(self, attribute: Attribute) -> None:
        name = attribute.name

        filename: str = self.graph_directory + "/" + name + ".json"

        with open(filename) as f:
            graph_json = json.load(f)

        graph = json_graph.node_link_graph(graph_json)

        self.set_graph(attribute, graph)

        return

if __name__ == '__main__':
    tax = Taxonomie()
    #tax.load_from_database()
    #tax.safe_to_file(Attribute.gender)
    tax.load_from_file(Attribute.gender)
    tax.plot(Attribute.gender)
    del tax
