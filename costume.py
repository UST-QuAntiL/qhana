class Costume:
    def __init__(self):
        self.id: int = 0
        self.dominant_color: str = ""
        self.dominant_condition: str = ""
        self.dominant_traits: [str] = []
        self.gender: str = ""
        self.dominant_age_impression: str = ""
        self.genres: [str] = []
    
    def print(self):
        print("Costume number: " + self.id)
        print("Dominant color: " + self.dominant_color)
        print("Dominant condition: " + self.dominant_condition)
        print("Dominant traits: ", end = '')
        if len(self.dominant_traits) > 0:
            print(self.dominant_traits[0])
        for i in range(1, len(self.dominant_traits)):
            print(", " + self.dominant_traits[i], end = '')
        print()
        print("Gender: " + self.gender)
        print("dominant age impression: " + self.dominant_age_impression)
        print("Genres: ", end = '')
        if len(self.genres) > 0:
            print(self.genres[0])
        for i in range(1, len(self.genres)):
            print(", " + self.genres[i], end = '')
        print()
