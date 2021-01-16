"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import asyncio
import matplotlib.pyplot as plt
import networkx as nx
from taxonomyLoadingService import TaxonomyLoadingService
from entityLoadingService import EntityLoadingService


async def main():
    tax_srv = TaxonomyLoadingService()
    location_tax = tax_srv.load('Profession')
    # nx.draw(location_tax.graph)
    # plt.show()

    ent_srv = EntityLoadingService()
    entities = ent_srv.load_entities('taxonomies/subset.csv')
    for entity in entities:
        print(entity)

if __name__ == "__main__":
    asyncio.run(main())
