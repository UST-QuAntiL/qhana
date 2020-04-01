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
    times_of_play = 10
    role = 11
    location_occurrence = 12
    material = 13

    # additional attributes
    type_of_basic_element = 14
    design = 15
    color_concept = 16