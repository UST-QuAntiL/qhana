from typing import Any
from typing import Dict
import networkx as nx
from networkx import Graph
from .database import Database
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os
from networkx.readwrite import json_graph
import simplejson as json
from .attribute import Attribute
from .singleton import Singleton
from .logger import Logger

"""
Represents a taxonomie
"""
class Taxonomie:
    """
    Initializes the taxonomie given the attribute and the graph
    """
    def __init__(self, attribute: Attribute, graph: Graph) -> None:
        self.attribute = Attribute
        self.graph = graph
        self.name = Attribute.get_name(attribute)
        return

    """
    Creates a taxonomie object from the database given the attribute
    """
    @staticmethod
    def create_from_db(attribute: Attribute, database: Database):
        graph = None

        # take the values from database if available
        if Attribute.get_database_table_name(attribute) is not None:
            graph = database.get_graph(Attribute.get_database_table_name(attribute))
        # if not, create the graph here
        else:
            graph = nx.DiGraph()
            if attribute == Attribute.geschlecht:
                graph.add_node("Geschlecht")
                graph.add_node("weiblich")
                graph.add_node("weiblich")
                graph.add_edge("Geschlecht", "weiblich")
                graph.add_edge("Geschlecht", "männlich")
            elif attribute == Attribute.ortsbegebenheit:
                graph.add_node("Ortsbegebenheit")
                graph.add_node("drinnen")
                graph.add_node("draußen")
                graph.add_node("drinnen und draußen")
            elif attribute == Attribute.farbeindruck:
                graph.add_node("Farbeindruck")
                graph.add_node("glänzend")
                graph.add_node("kräftig")
                graph.add_node("neon")
                graph.add_node("normal")
                graph.add_node("pastellig")
                graph.add_node("stumpf")
                graph.add_node("transparent")
            elif attribute == Attribute.koerperteil:
                graph.add_node("Körperteil")
                graph.add_node("Bein")
                graph.add_node("Fuß")
                graph.add_node("Ganzkörper")
                graph.add_node("Hals")
                graph.add_node("Hand")
                graph.add_node("Kopf")
                graph.add_node("Oberkörper")
                graph.add_node("Taille")
            elif attribute == Attribute.materialeindruck:
                graph.add_node("Materialeindruck")
                graph.add_node("anschmiegsam")
                graph.add_node("fest")
                graph.add_node("flauschig")
                graph.add_node("fließend")
                graph.add_node("künstlich")
                graph.add_node("leicht")
                graph.add_node("Oberkörper")
                graph.add_node("natürlich")
                graph.add_node("normal")
                graph.add_node("schwer")
                graph.add_node("steif")
                graph.add_node("weich")
            else:
                Logger.error("\"" + str(attribute) + "\" is a unknown attribute")
        
        taxonomie = Taxonomie(attribute, graph)
        return taxonomie

    """
    Creates a taxonomie object from the database given the attribute.
    If file or directory is None, the default will be used 
    """
    @staticmethod
    def create_from_json(attribute: Attribute, directory: str = "taxonomies", file: str = None):
        name_with_extension = (Attribute.get_name(attribute) + ".json") if file is None else file
        file_name = directory + "/" + name_with_extension

        with open(filename) as file_object:
            graph_json = json.load(file_object)

        graph = json_graph.node_link_graph(graph_json)

        return Taxonomie(attribute, graph)


    """ 
    Safe the json representing the taxonomie
    if no name is specified, the name of the
    taxonomie will be used
    """
    def save_json(self, directory: str = "taxonomies", file: str = None) -> None:
        if os.path.isdir(directory) == False:
            os.mkdir(directory)

        file_name = (self.name + ".json") if file is None else file
        file_path = directory + "/" + file_name
        graph_json = json_graph.node_link_data(self.graph)
        json.dump(graph_json, open(file_name,'w'), indent = 2)

        return

    """
    Safe the svg and dot representing the taxonomie
    if no name is specified, the name of the
    taxonomie will be used
    """
    def save_plot(self, directory: str = "taxonomies", name: str = None, display: bool = False) -> None:
        if os.path.isdir(directory) == False:
            os.mkdir(directory)

        file_name = self.name if name is None else name
        file_path_without_extension = directory + "/" + file_name
        file_path_svg = file_path_without_extension + ".svg"
        file_path_dot = file_path_without_extension + ".dot"

        nx.nx_agraph.write_dot(self.graph, file_path_dot)

        subprocess.call(["dot", "-Tsvg", file_path_dot, "-o", file_path_svg])

        absolute_file_path_svg = os.getcwd() + "/" + file_path_svg

        if display == True:
            os.startfile(absolute_file_path_svg, 'open')

        return

    """
    Check if element is in the taxonomie graph
    """
    def contains(self, element: Any) -> bool:
        return self.graph.has_node(element)

    """
    Gets the root of the given graph (tree)
    """
    def get_root(self, attribute: Attribute) -> str:
        return [n for n,d in self.graph.in_degree() if d == 0][0]
