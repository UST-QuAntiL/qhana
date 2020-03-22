from backend.costume import Costume
import backend.elementComparer as elemcomp
import backend.attributeComparer as attrcomp
import backend.aggregator as aggre
import argparse
from backend.costumeComparer import CostumeComparer
from backend.database import Database
from backend.taxonomie import Taxonomie
from backend.attribute import Attribute
from backend.logger import Logger, LogLevel

def parse_args():
    # Create one main parser
    parser = argparse.ArgumentParser(description='PlanQK - Machine Learning with Taxonomies')
    subparsers = parser.add_subparsers(required=True)

    # Add one sub parser for each task, i.e. printing (utils), clustering, ...
    utils_parser = subparsers.add_parser('utils')
    clustering_parser = subparsers.add_parser('clustering')

    # All arguments for global parser
    parser.add_argument('--log_level',
        dest='log_level',
        help='log level for the current session: 0 - Nothing, 1 - Errors [default], 2 - Warnings, 3 - Debug',
        default=1,
        type=int
    )

    # All arguments for utils parser

    # All arguments for clustering parser

    args = parser.parse_args()
    return args

def print_costumes(costumes: [Costume]) -> None:
    for i in range(1, len(costumes)):
        print(str(i) + ": " + str(costumes[i]))

def main() -> None:
    #args = parse_args()

    # Initialize logger
    Logger.initialize(LogLevel(1))

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