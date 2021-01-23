"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import asyncio
import matplotlib.pyplot as plt
import networkx as nx
from taxonomyLoadingService import TaxonomyLoadingService
from entityLoadingService import EntityLoadingService
from calculation.entityComparer import EntityComparer


async def main():
    tax_srv = TaxonomyLoadingService()
    # location_tax = tax_srv.load('Gender')
    # nx.draw(location_tax.graph)
    # plt.show()

    ent_srv = EntityLoadingService()
    entities = ent_srv.load_entities('taxonomies/subset.csv')
    attributes = ['DominantColor', 'DominantCondition', 'DominantCharacterTrait']

    # create comparer based on all attributes we want to use
    # for calculation and the given aggregator and transformer.
    comparer = EntityComparer()
    for attribute in attributes:
        comparer.add_attribute_comparer(attribute)

    # perform comparison
    result = comparer.calculate_pairwise_distance(entities)


if __name__ == "__main__":
    asyncio.run(main())
