from backend.costume import Costume
import backend.elementComparer as elemcomp
import backend.attributeComparer as attrcomp
import backend.aggregator as aggre
from backend.costumeComparer import CostumeComparer
from backend.database import Database
from backend.taxonomie import Taxonomie
from backend.attribute import Attribute
from backend.logger import Logger, LogLevel

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def main() -> None:
    # Initialize logger
    Logger.initialize(LogLevel.Debug)

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