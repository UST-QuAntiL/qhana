from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import networkx as nx
from networkx import Graph
from backend.database import Database
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os
from networkx.readwrite import json_graph
import simplejson as json
from backend.singleton import Singleton
from backend.logger import Logger
from enum import Enum

"""
Represents a taxonomie type, i.e. all possible
taxonomies in the database. This should not be
confused with the Attribute, which describes
any special property of an entity with which
is comparable.
"""
class TaxonomieType(Enum):
    alterseindruck = "alterseindruck"
    basiselement = "basiselement"
    charaktereigenschaft = "charaktereigenschaft"
    design = "design"
    farbeindruck = "farbeindruck"
    farbe = "farbe"
    farbkonzept = "farbkonzept"
    form = "form"
    funktion = "funktion"
    genre = "genre"
    koerpermodifikation = "koerpermodifikation"
    koerperteil = "koerperteil"
    material = "material"
    materialeindruck = "materialeindruck"
    operator = "operator"
    produktionsort = "produktionsort"
    rollenberuf = "rollenberuf"
    spielortdetail = "spielortdetail"
    spielort = "spielort"
    spielzeit = "spielzeit"
    stereotyp = "stereotyp"
    tageszeit = "tageszeit"
    teilelement = "teilelement"
    trageweise = "trageweise"
    typus = "typus"
    zustand = "zustand"
    geschlecht = "geschlecht"
    ortsbegebenheit = "ortsbegebenheit"
    stereotypRelevant = "stereotypRelevant"
    rollenrelevanz = "rollenrelevanz"

    @staticmethod
    def get_name(taxonomieType) -> str:
        if taxonomieType == taxonomieType.alterseindruck:
            return "Alterseindruck"
        elif taxonomieType == taxonomieType.basiselement:
            return "Basiselement"
        elif taxonomieType == taxonomieType.charaktereigenschaft:
            return "Charaktereigenschaft"
        elif taxonomieType == taxonomieType.design:
            return "Design"
        elif taxonomieType == taxonomieType.farbeindruck:
            return "Farbeindruck"
        elif taxonomieType == taxonomieType.farbe:
            return "Farbe"
        elif taxonomieType == taxonomieType.farbkonzept:
            return "Farbkonzept"
        elif taxonomieType == taxonomieType.form:
            return "Form"
        elif taxonomieType == taxonomieType.funktion:
            return "Funktion"
        elif taxonomieType == taxonomieType.genre:
            return "Genre"
        elif taxonomieType == taxonomieType.koerpermodifikation:
            return "Körpermodifikation"
        elif taxonomieType == taxonomieType.koerperteil:
            return "Köerperteil"
        elif taxonomieType == taxonomieType.material:
            return "Material"
        elif taxonomieType == taxonomieType.materialeindruck:
            return "Materialeindruck"
        elif taxonomieType == taxonomieType.operator:
            return "Operator"
        elif taxonomieType == taxonomieType.produktionsort:
            return "Produktionsort"
        elif taxonomieType == taxonomieType.rollenberuf:
            return "Rollenberuf"
        elif taxonomieType == taxonomieType.spielortdetail:
            return "Spielortdetail"
        elif taxonomieType == taxonomieType.spielort:
            return "Spielort"
        elif taxonomieType == taxonomieType.spielzeit:
            return "Spielzeit"
        elif taxonomieType == taxonomieType.stereotyp:
            return "Stereotyp"
        elif taxonomieType == taxonomieType.tageszeit:
            return "Tageszeit"
        elif taxonomieType == taxonomieType.teilelement:
            return "Teilelement"
        elif taxonomieType == taxonomieType.trageweise:
            return "Trageweise"
        elif taxonomieType == taxonomieType.typus:
            return "Typus"
        elif taxonomieType == taxonomieType.zustand:
            return "Zustand"
        elif taxonomieType == taxonomieType.geschlecht:
            return "Geschlecht"
        elif taxonomieType == taxonomieType.ortsbegebenheit:
            return "Ortsbegebenheit"
        elif taxonomieType == taxonomieType.stereotypRelevant:
            return "StereotypRelevant"
        elif taxonomieType == taxonomieType.rollenrelevanz:
            return "Rollenrelevanz"
        else:
            Logger.error("No name for taxonomieType \"" + str(taxonomieType) + "\" specified")
            raise ValueError("No name for taxonomieType \"" + str(taxonomieType) + "\" specified")
        return

    @staticmethod
    def get_database_table_name(taxonomieType) -> str:
        if taxonomieType == taxonomieType.alterseindruck:
            return "alterseindruckdomaene"
        elif taxonomieType == taxonomieType.basiselement:
            return "basiselementdomaene"
        elif taxonomieType == taxonomieType.charaktereigenschaft:
            return "charaktereigenschaftsdomaene"
        elif taxonomieType == taxonomieType.design:
            return "designdomaene"
        elif taxonomieType == taxonomieType.farbeindruck:
            return None
        elif taxonomieType == taxonomieType.farbe:
            return "farbendomaene"
        elif taxonomieType == taxonomieType.farbkonzept:
            return "farbkonzeptdomaene"
        elif taxonomieType == taxonomieType.form:
            return "formendomaene"
        elif taxonomieType == taxonomieType.funktion:
            return "funktionsdomaene"
        elif taxonomieType == taxonomieType.genre:
            return "genredomaene"
        elif taxonomieType == taxonomieType.koerpermodifikation:
            return "koerpermodifikationsdomaene"
        elif taxonomieType == taxonomieType.koerperteil:
            return None
        elif taxonomieType == taxonomieType.material:
            return "materialdomaene"
        elif taxonomieType == taxonomieType.materialeindruck:
            return None
        elif taxonomieType == taxonomieType.operator:
            return "operatordomaene"
        elif taxonomieType == taxonomieType.produktionsort:
            return "produktionsortdomaene"
        elif taxonomieType == taxonomieType.rollenberuf:
            return "rollenberufdomaene"
        elif taxonomieType == taxonomieType.spielortdetail:
            return "spielortdetaildomaene"
        elif taxonomieType == taxonomieType.spielort:
            return "spielortdomaene"
        elif taxonomieType == taxonomieType.spielzeit:
            return "spielzeitdomaene"
        elif taxonomieType == taxonomieType.stereotyp:
            return "stereotypdomaene"
        elif taxonomieType == taxonomieType.tageszeit:
            return "tageszeitdomaene"
        elif taxonomieType == taxonomieType.teilelement:
            return "teilelementdomaene"
        elif taxonomieType == taxonomieType.trageweise:
            return "trageweisendomaene"
        elif taxonomieType == taxonomieType.typus:
            return "typusdomaene"
        elif taxonomieType == taxonomieType.zustand:
            return "zustandsdomaene"
        elif taxonomieType == taxonomieType.geschlecht:
            return None
        elif taxonomieType == taxonomieType.ortsbegebenheit:
            return None
        elif taxonomieType == taxonomieType.stereotypRelevant:
            return None
        elif taxonomieType == taxonomieType.rollenrelevanz:
            return None 
        else:
            Logger.error("No name for taxonomieType \"" + str(taxonomieType) + "\" specified")
            raise ValueError("No name for taxonomieType \"" + str(taxonomieType) + "\" specified")
        return

"""
Represents a taxonomie
"""
class Taxonomie:
    """
    Static taxonomie pool to not create taxonomies everytime.
    This dictionary has the format (TaxonomieType, Taxonomie)
    """
    taxonomiePool = {}

    """
    Initializes the taxonomie given the taxonomieType and the graph
    """
    def __init__(self, taxonomieType: TaxonomieType, graph: Graph) -> None:
        self.taxonomieType = taxonomieType
        self.graph = graph
        self.name = TaxonomieType.get_name(taxonomieType)
        return

    """
    Creates a taxonomie object from the database given the taxonomieType.
    If the taxonomie is already in the pool, it will be used from there instead.
    """
    @staticmethod
    def create_from_db(taxonomieType: TaxonomieType, database: Database):
        if taxonomieType in Taxonomie.taxonomiePool:
            return Taxonomie.taxonomiePool[taxonomieType]

        graph = None

        # take the values from database if available
        if TaxonomieType.get_database_table_name(taxonomieType) is not None:
            graph = Taxonomie.__get_graph(taxonomieType.get_database_table_name(taxonomieType), database)
        # if not, create the graph here
        else:
            graph = nx.DiGraph()
            if taxonomieType == taxonomieType.geschlecht:
                graph.add_node("Geschlecht")
                graph.add_node("weiblich")
                graph.add_node("weiblich")
                graph.add_edge("Geschlecht", "weiblich")
                graph.add_edge("Geschlecht", "männlich")
            elif taxonomieType == taxonomieType.ortsbegebenheit:
                graph.add_node("Ortsbegebenheit")
                graph.add_node("drinnen")
                graph.add_node("draußen")
                graph.add_node("drinnen und draußen")
                graph.add_edge("Ortsbegebenheit", "drinnen")
                graph.add_edge("Ortsbegebenheit", "draußen")
                graph.add_edge("Ortsbegebenheit", "drinnen und draußen")
            elif taxonomieType == taxonomieType.farbeindruck:
                graph.add_node("Farbeindruck")
                graph.add_node("glänzend")
                graph.add_node("kräftig")
                graph.add_node("neon")
                graph.add_node("normal")
                graph.add_node("pastellig")
                graph.add_node("stumpf")
                graph.add_node("transparent")
                graph.add_edge("Farbeindruck", "glänzend")
                graph.add_edge("Farbeindruck", "kräftig")
                graph.add_edge("Farbeindruck", "neon")
                graph.add_edge("Farbeindruck", "normal")
                graph.add_edge("Farbeindruck", "pastellig")
                graph.add_edge("Farbeindruck", "pastellig")
                graph.add_edge("Farbeindruck", "stumpf")
            elif taxonomieType == taxonomieType.koerperteil:
                graph.add_node("Körperteil")
                graph.add_node("Bein")
                graph.add_node("Fuß")
                graph.add_node("Ganzkörper")
                graph.add_node("Hals")
                graph.add_node("Hand")
                graph.add_node("Kopf")
                graph.add_node("Oberkörper")
                graph.add_node("Taille")
                graph.add_edge("Körperteil", "Bein")
                graph.add_edge("Körperteil", "Fuß")
                graph.add_edge("Körperteil", "Ganzkörper")
                graph.add_edge("Körperteil", "Hals")
                graph.add_edge("Körperteil", "Hand")
                graph.add_edge("Körperteil", "Kopf")
                graph.add_edge("Körperteil", "Oberkörper")
                graph.add_edge("Körperteil", "Taille")
            elif taxonomieType == taxonomieType.materialeindruck:
                graph.add_node("Materialeindruck")
                graph.add_node("anschmiegsam")
                graph.add_node("fest")
                graph.add_node("flauschig")
                graph.add_node("fließend")
                graph.add_node("künstlich")
                graph.add_node("leicht")
                graph.add_node("natürlich")
                graph.add_node("normal")
                graph.add_node("schwer")
                graph.add_node("steif")
                graph.add_node("weich")
                graph.add_edge("Materialeindruck", "anschmiegsam")
                graph.add_edge("Materialeindruck", "fest")
                graph.add_edge("Materialeindruck", "flauschig")
                graph.add_edge("Materialeindruck", "fließend")
                graph.add_edge("Materialeindruck", "künstlich")
                graph.add_edge("Materialeindruck", "leicht")
                graph.add_edge("Materialeindruck", "natürlich")
                graph.add_edge("Materialeindruck", "normal")
                graph.add_edge("Materialeindruck", "schwer")
                graph.add_edge("Materialeindruck", "steif")
                graph.add_edge("Materialeindruck", "weich")
            elif taxonomieType == taxonomieType.stereotypRelevant:
                graph.add_node("StereotypRelevant")
                graph.add_node("ja")
                graph.add_node("nein")
                graph.add_edge("StereotypRelevant", "ja")
                graph.add_edge("StereotypRelevant", "nein")
            elif taxonomieType == taxonomieType.rollenrelevanz:
                graph.add_node("Rollenrelevanz")
                graph.add_node("Hauptrolle")
                graph.add_node("Nebenrolle")
                graph.add_node("Statist")
                graph.add_edge("Rollenrelevanz", "Hauptrolle")
                graph.add_edge("Rollenrelevanz", "Nebenrolle")
                graph.add_edge("Rollenrelevanz", "Statist")
            else:
                Logger.error("\"" + str(taxonomieType) + "\" is a unknown taxonomieType")
        
        taxonomie = Taxonomie(taxonomieType, graph)
        return taxonomie

    """
    Creates a taxonomie object from the database given the taxonomieType.
    If file or directory is None, the default will be used.
    If the taxonomie is already in the pool, it will be used from there instead.
    """
    @staticmethod
    def create_from_json(taxonomieType: TaxonomieType, directory: str = "taxonomies", file: str = None):
        if taxonomieType in Taxonomie.taxonomiePool:
            return Taxonomie.taxonomiePool[taxonomieType]

        name_with_extension = (taxonomieType.get_name(taxonomieType) + ".json") if file is None else file
        file_name = directory + "/" + name_with_extension

        with open(filename) as file_object:
            graph_json = json.load(file_object)

        graph = json_graph.node_link_graph(graph_json)

        return Taxonomie(taxonomieType, graph)

    """
    Returns the table of a taxonomie.
    In the kostuemrepo a taxonomie table is a "domaene" table with
    the structure "Child", "Parent"
    """
    @staticmethod
    def __get_taxonomie_table(name: str, database: Database) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        cursor = database.get_cursor()
        cursor.execute("SELECT * FROM " + name)
        rows = cursor.fetchall()
        cursor.close()
        return rows

    """
    Returns the graph object, i.e. the root node,
    given the name of the taxonomie table
    """
    @staticmethod
    def __get_graph(name: str, database: Database) -> Graph:
        nodes = {}
        # Format is (child, parent)
        rows = Taxonomie.__get_taxonomie_table(name, database)
        graph = nx.DiGraph()
        root_node = None

        for row in rows:
            child = row[0]
            parent = row[1]

            # If parent is none, this is the root node
            if parent is None:
                root_node = child
                nodes[child] = root_node
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
            Logger.error("No root node found in taxonomie")
            raise Exception("No root node found in taxonomie")

        if nx.algorithms.tree.recognition.is_tree(graph) == False:
            Logger.warning(name + " do not correspond to a tree")

        return graph

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
        json.dump(graph_json, open(file_path,'w'), indent = 2)

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
    def get_root(self, taxonomieType: TaxonomieType) -> str:
        return [n for n,d in self.graph.in_degree() if d == 0][0]
