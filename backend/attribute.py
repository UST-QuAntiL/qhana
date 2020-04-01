import enum

class Attribute(enum.Enum):
    # these are the attributes for the simple model
    color = 1
    traits = 2
    condition = 3
    stereotype = 4
    gender = 5
    age_impression = 6
    genre = 7

    # these are the attributes for the extended model
    job = 8
    times_of_day = 9