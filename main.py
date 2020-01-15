from costume import Costume
from database import Database
from taxonomie import Taxonomie
from typing import Any
from attribute import Attribute

def aggregate(single_values: [float]) -> float:
    result = 0.0

    for value in single_values:
        result += value
    
    return result / len(single_values)

def compare(first: Costume, second: Costume) -> float:
    # compare each attribute individually
    tax = Taxonomie()
    tax.load_all_from_file()

    aggregation_values = []

    #aggregation_values.append(tax.wu_palmer(Attribute.color, first.dominant_color, second.dominant_color))
    #aggregation_values.append(tax.wu_palmer_set(Attribute.traits, first.dominant_traits, second.dominant_traits))
    #aggregation_values.append(tax.wu_palmer(Attribute.condition, first.dominant_condition, second.dominant_condition))
    #aggregation_values.append(tax.wu_palmer_set(Attribute.stereotype, first.stereotypes, second.stereotypes))
    #aggregation_values.append(tax.wu_palmer(Attribute.gender, first.gender, second.gender))
    #aggregation_values.append(tax.wu_palmer(Attribute.age_impression, first.dominant_age_impression, second.dominant_age_impression))
    aggregation_values.append(tax.wu_palmer_set(Attribute.genre, first.genres, second.genres))

    return aggregate(aggregation_values)


# Establish connection to db
db = Database()
db.open()

costumes = db.get_costumes()
print(costumes[0])
print(costumes[1])
print("Similarity = " + str(compare(costumes[0], costumes[1])))
