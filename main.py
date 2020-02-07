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

    tax = Taxonomie()
    tax.load_all()
    tax.plot_all(False)

    # Establish connection to db
    db = Database()
    db.open()

    costumes = db.get_costumes()
    costumeComparer = CostumeComparer()

    first = 3087
    second = 3090

    comparedResult = costumeComparer.compare_distance(costumes[first], costumes[second])

    print(costumes[first])
    print(costumes[second])
    print("Compared result: " + str(round(comparedResult, 2)))

    #print_costumes(costumes)

    return

if __name__== "__main__":
    main()