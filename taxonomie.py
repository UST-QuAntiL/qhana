from typing import Any
import networkx as nx
from networkx import Graph
from database import Database
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import subprocess
import os

class Taxonomie():
    def __init__(self) -> None:
        self.database: Database = Database()
        self.is_loaded: bool = False
        self.plot_datatype: str = "png"
        self.plot_directory: str = "plots"
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
    def load(self) -> None:
        self.database.open()
        self.color = self.database.get_color()
        self.traits = self.database.get_traits()
        self.condition = self.database.get_condition()
        self.stereotype = self.database.get_stereotype()
        self.gender = self.database.get_gender()
        self.age_impression = self.database.get_age_impression()
        self.genre = self.database.get_genre()
        self.is_loaded = True

    # Gets the root of the given graph (tree)
    def get_root(self, graph: Graph) -> str:
        return [n for n,d in graph.in_degree() if d == 0][0]

    # Safe the given graph as dot and png file and
    # displays the created image
    def plot(self, graph: Graph, display: bool = True, name: str = "") -> None:
        if self.is_loaded == False:
            print("Taxonomie is not loaded")
            return

        if os.path.isdir(self.plot_directory) == False:
            os.mkdir(self.plot_directory)

        if name == "":
            name = self.get_root(graph)

        filepath: str = self.plot_directory + "/" + name

        nx.nx_agraph.write_dot(graph, filepath + ".dot")
        subprocess.call(["dot", "-T" + self.plot_datatype, filepath + ".dot", "-o", filepath + "." + self.plot_datatype])

        if display == True:
            img = Image.open(filepath + "." + self.plot_datatype)
            img.show()
        
        return
    
if __name__ == '__main__':
    tax = Taxonomie()
    tax.load()
    tax.plot(tax.traits)
    del tax
