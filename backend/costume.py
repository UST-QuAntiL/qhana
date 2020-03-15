from typing import Any
from attribute import Attribute

# This class represents a costume in the simplified model
# NOTE: We use stereotyp although it is not defined in the simplified model

class Costume:
    def __init__(self) -> None:
        self.dominant_color: str = ""
        self.dominant_traits: [str] = []
        self.dominant_condition: str = ""
        self.stereotypes: [str] = []
        self.gender: str = ""
        self.dominant_age_impression: str = ""
        self.genres: [str] = []
    
    # Returns the costume in a single string
    def __str__(self) -> str:
        output = "Costume: "
        output += str(self.dominant_color) + ", "
        output += str(self.dominant_condition) + ", "
        output += str(self.dominant_traits) + ", "
        output += str(self.stereotypes) + ", "
        output += str(self.gender) + ", "
        output += str(self.dominant_age_impression) + ", "
        output += str(self.genres)
        return output

    def print(self) -> None:
        print("Dominant color: " + self.dominant_color)
        print("Dominant condition: " + self.dominant_condition)
        print("Dominant traits: ", end = '')
        if len(self.dominant_traits) > 0:
            print(self.dominant_traits[0])
        for i in range(1, len(self.dominant_traits)):
            print(", " + self.dominant_traits[i], end = '')
        print()
        print("Stereotypes: ", end = '')
        if len(self.stereotypes) > 0:
            print(self.stereotypes[0])
        for i in range(1, len(self.stereotypes)):
            print(", " + self.stereotypes[i], end = '')
        print()
        print("Gender: " + self.gender)
        print("dominant age impression: " + self.dominant_age_impression)
        print("Genres: ", end = '')
        if len(self.genres) > 0:
            print(self.genres[0])
        for i in range(1, len(self.genres)):
            print(", " + self.genres[i], end = '')
        print()
