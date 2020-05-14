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
    ortsbegebenheit = "ortsbegebenheit"
    dominanteFarbe = "dominanteFarbe"
    stereotypRelevant = "stereotypRelevant"
    dominanteFunktion = "dominanteFunktion"
    dominanterZustand = "dominanterZustand"
    dominanteCharaktereigenschaft = "dominanteCharaktereigenschaft"
    stereotyp = "stereotyp"
    geschlecht = "geschlecht"
    dominanterAlterseindruck = "dominanterAlterseindruck"
    genre = "genre"
    rollenberuf = "rollenberuf"
    dominantesAlter = "dominantesAlter"
    rollenrelevanz = "rollenrelevanz"
    spielzeit = "spielzeit"
    tageszeit = "tageszeit"
    koerpermodifikation = "koerpermodifikation"
    kostuemZeit = "kostuemZeit"
    familienstand = "familienstand"
    charaktereigenschaft = "charaktereigenschaft"
    spielort = "spielort"
    spielortDetail = "spielortDetail"
    alterseindruck = "alterseindruck"
    alter = "alter"
    basiselement = "basiselement"
    design = "design"
    form = "form"
    trageweise = "trageweise"
    zustand = "zustand"

    """
    Returns the human representable name for the
    given Attribute.
    """
    @staticmethod
    def get_name(attribute) -> str:
        if attribute == Attribute.ortsbegebenheit:
            return "Ortsbegebenheit"
        elif attribute == Attribute.dominanteFarbe:
            return "Dominante Farbe"
        elif attribute == Attribute.stereotypRelevant:
            return "Stereotyp relevant"
        elif attribute == Attribute.dominanteFunktion:
            return "Dominante Funktion"
        elif attribute == Attribute.dominanterZustand:
            return "Dominanter Zustand"
        elif attribute == Attribute.dominanteCharaktereigenschaft:
            return "Dominante Charaktereigenschaft"
        elif attribute == Attribute.stereotyp:
            return "Stereotyp"
        elif attribute == Attribute.geschlecht:
            return "Geschlecht"
        elif attribute == Attribute.dominanterAlterseindruck:
            return "Dominanter Alterseindruck"
        elif attribute == Attribute.genre:
            return "Genre"
        elif attribute == Attribute.rollenberuf:
            return "Rollenberuf"
        elif attribute == Attribute.dominantesAlter:
            return "Dominantes Alter"
        elif attribute == Attribute.rollenrelevanz:
            return "Rollenrelevanz"
        elif attribute == Attribute.spielzeit:
            return "Spielzeit"
        elif attribute == Attribute.tageszeit:
            return "Tageszeit"
        elif attribute == Attribute.koerpermodifikation:
            return "Körpermodifikation"
        elif attribute == Attribute.kostuemZeit:
            return "Kostümzeit"
        elif attribute == Attribute.familienstand:
            return "Familienstand"
        elif attribute == Attribute.charaktereigenschaft:
            return "Charaktereigenschaft"
        elif attribute == Attribute.spielort:
            return "Spielort"
        elif attribute == Attribute.spielortDetail:
            return "SpielortDetail"
        elif attribute == Attribute.alterseindruck:
            return "Alterseindruck"
        elif attribute == Attribute.alter:
            return "Alter"
        elif attribute == Attribute.basiselement:
            return "Basiselement"
        elif attribute == Attribute.design:
            return "Design"
        elif attribute == Attribute.form:
            return "Form"
        elif attribute == Attribute.trageweise:
            return "Trageweise"
        elif attribute == Attribute.zustand:
            return "Zustand"
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
        if attribute == Attribute.ortsbegebenheit:
            return TaxonomieType.ortsbegebenheit
        elif attribute == Attribute.dominanteFarbe:
            return TaxonomieType.farbe
        elif attribute == Attribute.stereotypRelevant:
            return TaxonomieType.stereotypRelevant
        elif attribute == Attribute.dominanteFunktion:
            return TaxonomieType.funktion
        elif attribute == Attribute.dominanterZustand:
            return TaxonomieType.zustand
        elif attribute == Attribute.dominanteCharaktereigenschaft:
            return TaxonomieType.typus
        elif attribute == Attribute.stereotyp:
            return TaxonomieType.stereotyp
        elif attribute == Attribute.geschlecht:
            return TaxonomieType.geschlecht
        elif attribute == Attribute.dominanterAlterseindruck:
            return TaxonomieType.alterseindruck
        elif attribute == Attribute.genre:
            return TaxonomieType.genre
        elif attribute == Attribute.rollenberuf:
            return TaxonomieType.rollenberuf
        elif attribute == Attribute.dominantesAlter:
            return None
        elif attribute == Attribute.rollenrelevanz:
            return TaxonomieType.rollenrelevanz
        elif attribute == Attribute.spielzeit:
            return TaxonomieType.spielzeit
        elif attribute == Attribute.tageszeit:
            return TaxonomieType.tageszeit
        elif attribute == Attribute.koerpermodifikation:
            return TaxonomieType.koerpermodifikation
        elif attribute == Attribute.kostuemZeit:
            return None
        elif attribute == Attribute.familienstand:
            return TaxonomieType.familienstand
        elif attribute == Attribute.charaktereigenschaft:
            return TaxonomieType.charaktereigenschaft
        elif attribute == Attribute.spielort:
            return TaxonomieType.spielort
        elif attribute == Attribute.spielortDetail:
            return TaxonomieType.spielortDetail
        elif attribute == Attribute.alterseindruck:
            return TaxonomieType.alterseindruck
        elif attribute == Attribute.alter:
            return None
        elif attribute == Attribute.basiselement:
            return TaxonomieType.basiselement
        elif attribute == Attribute.design:
            return TaxonomieType.design
        elif attribute == Attribute.form:
            return TaxonomieType.form
        elif attribute == Attribute.trageweise:
            return TaxonomieType.trageweise
        elif attribute == Attribute.zustand:
            return TaxonomieType.zustand
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
        if attribute == Attribute.dominantesAlter:
            return None
        elif attribute == Attribute.kostuemZeit:
            return None
        elif attribute == Attribute.alter:
            return None
        return Taxonomie.create_from_db(Attribute.get_taxonomie_type(attribute), database)
