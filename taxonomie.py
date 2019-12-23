from typing import Callable
from typing import List

class Taxonomie():
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.children: List[Taxonomie] = []
    
    # Returns the taxonomie as a string formated with the format lambda
    def __str__(self, level: int = 0, format: Callable[[int, str], str] = lambda level, name: "\t" * level + "<" + name + ">" + "\n") -> str:
        output: str = format(level, self.name)
        for child in self.children:
            output += child.__str__(level + 1)
        return output
    
    def add_child(self, child):
        self.children.append(child)
