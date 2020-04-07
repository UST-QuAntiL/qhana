import enum
from typing import Any
from backend.taxonomie import Taxonomie, TaxonomieType
from backend.logger import Logger
from backend.database import Database

"""
This class is an enum for all attributes.
An attribute is a special property that is used
to describe one characteristic of an entity.
For example, a costume has the attribute
"Dominante Farbe". The attribute should not be
confused with TaxonomieType, which is the enum
for all possible taxonomies in the database.
"""
class Attribute(enum.Enum):
    dominanteFarbe = 1
    dominanteCharaktereigenschaft = 2
    dominanterZustand = 3
    stereotyp = 4
    geschlecht = 5
    dominanterAlterseindruck = 6
    genre = 7

    """
    Returns the human representable name for the
    given Attribute.
    """
    @staticmethod
    def get_name(attribute) -> str:
        if attribute == Attribute.dominanteFarbe:
            return "Dominante Farbe"
        elif attribute == Attribute.dominanteCharaktereigenschaft:
            return "Dominante Charaktereigenschaft"
        elif attribute == Attribute.dominanterZustand:
            return "Dominanter Zustand"
        elif attribute == Attribute.stereotyp:
            return "Stereotyp"
        elif attribute == Attribute.geschlecht:
            return "Geschlecht"
        elif attribute == Attribute.dominanterAlterseindruck:
            return "Dominanter Alterseindruck"
        elif attribute == Attribute.genre:
            return "Genre"
        else:
            Logger.error("No name for attribute \"" + str(attribute) + "\" specified")
            raise ValueError("No name for attribute \"" + str(attribute) + "\" specified")
        return

    """
    Returns the corresponding taxonomie type
    this attribute is used for.
    Note, that an attribute has only one taxonomie type,
    while a taxonomie type can be used by multiple attributes.
    """
    @staticmethod
    def get_taxonomie_type(attribute) -> str:
        if attribute == Attribute.dominanteFarbe:
            return TaxonomieType.farbe
        elif attribute == Attribute.dominanteCharaktereigenschaft:
            return TaxonomieType.typus
        elif attribute == Attribute.dominanterZustand:
            return TaxonomieType.zustand
        elif attribute == Attribute.stereotyp:
            return TaxonomieType.stereotyp
        elif attribute == Attribute.geschlecht:
            return TaxonomieType.geschlecht
        elif attribute == Attribute.dominanterAlterseindruck:
            return TaxonomieType.alterseindruck
        elif attribute == Attribute.genre:
            return TaxonomieType.genre
        else:
            Logger.error("No taxonomie type for attribute \"" + str(attribute) + "\" specified")
            raise ValueError("No taxonomie type for attribute \"" + str(attribute) + "\" specified")
        return

    """
    Gets the base on which this attribute can be compared
    with others.
    """
    @staticmethod
    def get_base(attribute, database: Database) -> Any:
        return Taxonomie.create_from_db(Attribute.get_taxonomie_type(attribute), database)
