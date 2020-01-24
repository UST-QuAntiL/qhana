from costume import Costume
import elementComparer as elemcomp
import attributeComparer as attrcomp
import aggregator as aggre
from costumeComparer import CostumeComparer
from database import Database
from taxonomie import Taxonomie
from attribute import Attribute

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def main() -> None:
    # Establish connection to db
    db = Database()
    db.open()

    costumes = db.get_costumes()
    costumeComparer = CostumeComparer()

    comparedResult = costumeComparer.compare(costumes[len(costumes)-2], costumes[len(costumes)-1])

    print(costumes[len(costumes)-2])
    print(costumes[len(costumes)-1])
    print("Compared result: " + str(round(comparedResult, 2)))

    return

if __name__== "__main__":
    main()