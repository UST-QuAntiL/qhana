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
Represents all taxonomies from kostuem repository
"""
class Taxonomie(Singleton):
    def __init__(self) -> None:
        self.database: Database = Database()
        self.plot_datatype: str = "svg"
        self.plot_directory: str = "taxonomies"     # directory for svg and dot files
        self.graph_directory: str = "taxonomies"    # directory for json files

        # these are the taxonomies for the simple model
        self.color: Graph = None
        self.traits: Graph = None
        self.condition: Graph = None
        self.stereotype: Graph = None
        self.gender: Graph = None
        self.age_impression: Graph = None
        self.genre: Graph = None

        # these are the taxonomies for the extended model
        self.job: Graph = None
        self.times_of_day: Graph = None
        self.times_of_play: Graph = None
        self.role: Graph = None
        self.location_occurrence: Graph = None
        self.material: Graph = None

        # additional attributes
        self.type_of_basic_element: Graph = None
        self.design: Graph = None
        self.color_concept: Graph = None

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

    # Loads all taxonomies from file. If one file does not exist
    # load all taxonomies from database
    def load_all(self) -> None:
        db_connection_needed: bool = False
        filenames: Dict[Attribute, str] = {
            Attribute.color: self.graph_directory + "/" + Attribute.color.name + ".json",
            Attribute.traits: self.graph_directory + "/" + Attribute.traits.name + ".json",
            Attribute.condition: self.graph_directory + "/" + Attribute.condition.name + ".json",
            Attribute.stereotype: self.graph_directory + "/" + Attribute.stereotype.name + ".json",
            Attribute.gender: self.graph_directory + "/" + Attribute.gender.name + ".json",
            Attribute.age_impression: self.graph_directory + "/" + Attribute.age_impression.name + ".json",
            Attribute.genre: self.graph_directory + "/" + Attribute.genre.name + ".json",
            Attribute.job: self.graph_directory + "/" + Attribute.job.name + ".json",
            Attribute.times_of_day: self.graph_directory + "/" + Attribute.times_of_day.name + ".json",
            Attribute.times_of_play: self.graph_directory + "/" + Attribute.times_of_play.name + ".json",
            Attribute.role: self.graph_directory + "/" + Attribute.role.name + ".json",
            Attribute.location_occurrence: self.graph_directory + "/" + Attribute.location_occurrence.name + ".json",
            Attribute.material: self.graph_directory + "/" + Attribute.material.name + ".json",
            Attribute.type_of_basic_element:self.graph_directory + "/" + Attribute.type_of_basic_element.name + ".json",
            Attribute.design: self.graph_directory + "/" + Attribute.design.name + ".json",
            Attribute.color_concept: self.graph_directory + "/" + Attribute.color_concept.name + ".json"
        }

        for attribute in filenames:
            os.path.isfile(filenames[attribute])
            if os.path.isfile(filenames[attribute]) == False:
                db_connection_needed = True
        
        if db_connection_needed == True:
            self.load_all_from_database()
            Logger.debug("Load taxonomies from database")
        else:
            for attribute in filenames:
                self.load_from_file(attribute)
                Logger.debug("Load taxonomies from file")
        return

    # Loads the taxonomies from database
    def load_all_from_database(self) -> None:
        self.database.open()

        self.color = self.database.get_color()
        self.traits = self.database.get_traits()
        self.condition = self.database.get_condition()
        self.stereotype = self.database.get_stereotype()
        self.gender = self.database.get_gender()
        self.age_impression = self.database.get_age_impression()
        self.genre = self.database.get_genre()
        self.job = self.database.get_job()
        self.times_of_day = self.database.get_times_of_day()
        self.times_of_play = self.database.get_times_of_play()
        self.role = self.database.get_role()
        self.location_occurrence = self.database.get_location_occurrence()
        self.material = self.database.get_material()
        self.type_of_basic_element = self.database.get_type_of_basic_element()
        self.design = self.database.get_design()
        self.color_concept = self.database.get_color_concept()

        self.database.close()

        #self.all = self.merge(
        #    self.all_taxonomie_name,
        #    self.color,
        #    self.traits,
        #    self.condition,
        #    self.stereotype,
        #    self.gender,
        #    self.age_impression,
        #    self.genre)

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
        self.safe_to_file(Attribute.color)
        self.safe_to_file(Attribute.traits)
        self.safe_to_file(Attribute.condition)
        self.safe_to_file(Attribute.stereotype)
        self.safe_to_file(Attribute.gender)
        self.safe_to_file(Attribute.age_impression)
        self.safe_to_file(Attribute.genre)
        self.safe_to_file(Attribute.job)
        self.safe_to_file(Attribute.times_of_day)
        self.safe_to_file(Attribute.times_of_play)
        self.safe_to_file(Attribute.role)
        self.safe_to_file(Attribute.location_occurrence)
        self.safe_to_file(Attribute.material)
        self.safe_to_file(Attribute.type_of_basic_element)
        self.safe_to_file(Attribute.design)
        self.safe_to_file(Attribute.color_concept)

        Logger.normal("All taxonomies have been saved as json to " + self.graph_directory)

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
        self.load_from_file(Attribute.job)
        self.load_from_file(Attribute.times_of_day)
        self.load_from_file(Attribute.times_of_play)
        self.load_from_file(Attribute.role)
        self.load_from_file(Attribute.location_occurrence)
        self.load_from_file(Attribute.material)
        self.load_from_file(Attribute.type_of_basic_element)
        self.load_from_file(Attribute.design)
        self.load_from_file(Attribute.color_concept)

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
        elif attribute == Attribute.job:
            return self.job
        elif attribute == Attribute.times_of_day:
            return self.times_of_day
        elif attribute == Attribute.times_of_play:
            return self.times_of_play
        elif attribute == Attribute.role:
            return self.role
        elif attribute == Attribute.location_occurrence:
            return self.location_occurrence
        elif attribute == Attribute.material:
            return self.material
        elif attribute == Attribute.type_of_basic_element:
            return self.type_of_basic_element
        elif attribute == Attribute.design:
            return self.design
        elif attribute == Attribute.color_concept:
            return self.color_concept
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
        elif attribute == Attribute.job:
            self.job = graph
        elif attribute == Attribute.times_of_day:
            self.times_of_day = graph
        elif attribute == Attribute.times_of_play:
            self.times_of_play = graph
        elif attribute == Attribute.role:
            self.role = graph
        elif attribute == Attribute.location_occurrence:
            self.location_occurrence = graph
        elif attribute == Attribute.material:
            self.material = graph
        elif attribute == Attribute.type_of_basic_element:
            self.type_of_basic_element = graph
        elif attribute == Attribute.design:
            self.design = graph
        elif attribute == Attribute.color_concept:
            self.color_concept = graph
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

    # Check if the taxonomie contains the given element
    def contains(self, attribute: Attribute, element: Any) -> bool:
        return

    # Safe the graph to the given attribute as dot 
    # and image file and displays the created image
    def plot(self, attribute: Attribute, display: bool = True) -> None:
        if os.path.isdir(self.plot_directory) == False:
            os.mkdir(self.plot_directory)

        name = attribute.name

        filepath: str = self.plot_directory + "/" + name

        graph = self.get_graph(attribute)

        nx.nx_agraph.write_dot(graph, filepath + ".dot")
        subprocess.call(["dot", "-T" + self.plot_datatype, filepath + ".dot", "-o", filepath + "." + self.plot_datatype])

        filename = os.getcwd() + "/" + filepath + "." + self.plot_datatype

        if display == True:
            os.startfile(filename, 'open')

        return

    # Safe all graph to dot and image files
    def plot_all(self, display: bool = False) -> None:
        self.plot(Attribute.color, display)
        self.plot(Attribute.traits, display)
        self.plot(Attribute.condition, display)
        self.plot(Attribute.stereotype, display)
        self.plot(Attribute.gender, display)
        self.plot(Attribute.age_impression, display)
        self.plot(Attribute.genre, display)
        self.plot(Attribute.job, display)
        self.plot(Attribute.times_of_day, display)
        self.plot(Attribute.times_of_play, display)
        self.plot(Attribute.role, display)
        self.plot(Attribute.location_occurrence, display)
        self.plot(Attribute.material, display)
        self.plot(Attribute.type_of_basic_element, display)
        self.plot(Attribute.design, display)
        self.plot(Attribute.color_concept, display)

        Logger.normal("All taxonomies have been saved as dot and svg to " + self.plot_directory)
        return
