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
from attribute import Attribute

class Taxonomie():
    def __init__(self) -> None:
        self.database: Database = Database()
        self.plot_datatype: str = "svg"
        self.plot_directory: str = "plots"
        self.graph_directory: str = "graphs"
        self.color: Graph = None
        self.traits: Graph = None
        self.condition: Graph = None
        self.stereotype: Graph = None
        self.gender: Graph = None
        self.age_impression: Graph = None
        self.genre: Graph = None
        self.all_taxonomie_name = "Taxonomie"
        self.all: Graph = None
        return

    def __del__(self) -> None:
        del self.database
        return
    
    def merge(self, root_name: str, *graphs: [Graph]) -> Graph:
        graph = nx.DiGraph()
        graph.add_node(root_name)

        for subgraph in graphs:
            graph.add_edges_from(subgraph.edges(data = True))
            graph.add_nodes_from(subgraph.nodes(data = True))
            subgraph_root_name = [n for n,d in subgraph.in_degree() if d == 0][0]
            graph.add_edge(root_name, subgraph_root_name)

        return graph

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

        self.all = self.merge(
            self.all_taxonomie_name,
            self.color,
            self.traits,
            self.condition,
            self.stereotype,
            self.gender,
            self.age_impression,
            self.genre)

        return

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
            raise Exception("Unkown attribute \"" + str(attribute) + "\"")

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
            raise Exception("Unkown attribute \"" + attribute + "\"")
        return

    # Gets the root of the given graph (tree)
    def get_root(self, attribute: Attribute) -> str:
        graph = self.get_graph(attribute)
        return [n for n,d in graph.in_degree() if d == 0][0]

    # Gets the count of edges between first and second
    def get_count_of_edges(self, attribute: Attribute, first: Any, sedonc: Any) -> int:
        return 0

    # Safe the graph to the given attribute as dot 
    # and png file and displays the created image
    def plot(self, attribute: Attribute, display: bool = True) -> None:
        if os.path.isdir(self.plot_directory) == False:
            os.mkdir(self.plot_directory)

        name = attribute.name

        filepath: str = self.plot_directory + "\\" + name

        graph = self.get_graph(attribute)

        nx.nx_agraph.write_dot(graph, filepath + ".dot")
        subprocess.call(["dot", "-T" + self.plot_datatype, filepath + ".dot", "-o", filepath + "." + self.plot_datatype])

        filename = os.getcwd() + "\\" + filepath + "." + self.plot_datatype

        os.startfile(filename, 'open')

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

    # Safes all graphs to data files
    def safe_all_to_file(self) -> None:
        tax.safe_to_file(Attribute.color)
        tax.safe_to_file(Attribute.traits)
        tax.safe_to_file(Attribute.condition)
        tax.safe_to_file(Attribute.stereotype)
        tax.safe_to_file(Attribute.gender)
        tax.safe_to_file(Attribute.age_impression)
        tax.safe_to_file(Attribute.genre)
        return

    # Loads all graphs to data files
    def load_all_from_file(self) -> None:
        self.load_from_file(Attribute.color)
        self.load_from_file(Attribute.traits)
        self.load_from_file(Attribute.condition)
        self.load_from_file(Attribute.stereotype)
        self.load_from_file(Attribute.gender)
        self.load_from_file(Attribute.age_impression)
        self.load_from_file(Attribute.genre)
        return

if __name__ == '__main__':
    tax = Taxonomie()
    #tax.load_from_database()
    tax.load_all_from_file()
    #tax.safe_all_to_file()
    #tax.load_from_file(Attribute.color)
    tax.plot(Attribute.color)
    #print("Brauntöne - Orange: " + str(tax.wu_palmer_compare(Attribute.color, "Brauntöne", "Orange")))
    #print("Farben - Dunkelviolett: " + str(tax.wu_palmer_compare(Attribute.color, "Farben", "Dunkelviolett")))
    #print("Grüntöne - Farbe: " + str(tax.wu_palmer_compare(Attribute.color, "Grüntöne", "Farben")))

    del tax
