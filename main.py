from costume import Costume
import elementComparer as elemcomp
import attributeComparer as attrcomp
import aggregator as aggre
from costumeComparer import CostumeComparer
from database import Database

# Establish connection to db
db = Database()
db.open()

costumes = db.get_costumes()
#print(costumes[0])
#print(costumes[1])
#print("Similarity = " + str(compare(costumes[0], costumes[1])))

# Create comparer using all default values
costumeComparer = CostumeComparer()

for costume in costumes:
    print(costume)