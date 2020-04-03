import enum

class Attribute(enum.Enum):
    alterseindruck = 1
    basiselement = 2
    charaktereigenschaft = 3
    design = 4
    farbeindruck = 5
    farbe = 6
    farbkonzept = 7
    form = 8
    funktion = 9
    genre = 10
    koerpermodifikation = 11
    koerperteil = 12
    material = 13
    materialeindruck = 14
    operator = 15
    produktionsort = 16
    rollenberuf = 17
    spielortdetail = 18
    spielort = 19
    spielzeit = 20
    stereotyp = 21
    tageszeit = 22
    teilelement = 23
    trageweise = 24
    typus = 25
    zustand = 26
    geschlecht = 27
    ortsbegebenheit = 28

    @staticmethod
    def get_name(attribute) -> str:
        if attribute == Attribute.alterseindruck:
            return "Alterseindruck"
        elif attribute == Attribute.basiselement:
            return "Basiselement"
        elif attribute == Attribute.charaktereigenschaft:
            return "Charaktereigenschaft"
        elif attribute == Attribute.design:
            return "Design"
        elif attribute == Attribute.farbeindruck:
            return "Farbeindruck"
        elif attribute == Attribute.farbe:
            return "Farbe"
        elif attribute == Attribute.farbkonzept:
            return "Farbkonzept"
        elif attribute == Attribute.form:
            return "Form"
        elif attribute == Attribute.funktion:
            return "Funktion"
        elif attribute == Attribute.genre:
            return "Genre"
        elif attribute == Attribute.koerpermodifikation:
            return "Körpermodifikation"
        elif attribute == Attribute.koerperteil:
            return "Köerperteil"
        elif attribute == Attribute.material:
            return "Material"
        elif attribute == Attribute.materialeindruck:
            return "Materialeindruck"
        elif attribute == Attribute.operator:
            return "Operator"
        elif attribute == Attribute.produktionsort:
            return "Produktionsort"
        elif attribute == Attribute.rollenberuf:
            return "Rollenberuf"
        elif attribute == Attribute.spielortdetail:
            return "Spielortdetail"
        elif attribute == Attribute.spielort:
            return "Spielort"
        elif attribute == Attribute.spielzeit:
            return "Spielzeit"
        elif attribute == Attribute.stereotyp:
            return "Stereotyp"
        elif attribute == Attribute.tageszeit:
            return "Tageszeit"
        elif attribute == Attribute.teilelement:
            return "Teilelement"
        elif attribute == Attribute.trageweise:
            return "Trageweise"
        elif attribute == Attribute.typus:
            return "Typus"
        elif attribute == Attribute.zustand:
            return "Zustand"
        elif attribute == Attribute.geschlecht:
            return "Geschlecht"
        elif attribute == Attribute.ortsbegebenheit:
            return "Ortsbegebenheit"
        else:
            Logger.error("No name for attribute \"" + str(attribute) + "\" specified")
            raise ValueError("No name for attribute \"" + str(attribute) + "\" specified")
        return

    @staticmethod
    def get_database_table_name(attribute) -> str:
        if attribute == Attribute.alterseindruck:
            return "alterseindruckdomaene"
        elif attribute == Attribute.basiselement:
            return "basiselementdomaene"
        elif attribute == Attribute.charaktereigenschaft:
            return "charaktereigenschaftsdomaene"
        elif attribute == Attribute.design:
            return "designdomaene"
        elif attribute == Attribute.farbeindruck:
            return None
        elif attribute == Attribute.farbe:
            return "farbendomaene"
        elif attribute == Attribute.farbkonzept:
            return "farbkonzeptdomaene"
        elif attribute == Attribute.form:
            return "formendomaene"
        elif attribute == Attribute.funktion:
            return "funktionsdomaene"
        elif attribute == Attribute.genre:
            return "genredomaene"
        elif attribute == Attribute.koerpermodifikation:
            return "koerpermodifikationsdomaene"
        elif attribute == Attribute.koerperteil:
            return None
        elif attribute == Attribute.material:
            return "materialdomaene"
        elif attribute == Attribute.materialeindruck:
            return None
        elif attribute == Attribute.operator:
            return "operatordomaene"
        elif attribute == Attribute.produktionsort:
            return "produktionsortdomaene"
        elif attribute == Attribute.rollenberuf:
            return "rollenberufdomaene"
        elif attribute == Attribute.spielortdetail:
            return "spielortdetaildomaene"
        elif attribute == Attribute.spielort:
            return "spielortdomaene"
        elif attribute == Attribute.spielzeit:
            return "spielzeitdomaene"
        elif attribute == Attribute.stereotyp:
            return "stereotypdomaene"
        elif attribute == Attribute.tageszeit:
            return "tageszeitdomaene"
        elif attribute == Attribute.teilelement:
            return "teilelementdomaene"
        elif attribute == Attribute.trageweise:
            return "trageweisendomaene"
        elif attribute == Attribute.typus:
            return "typusdomaene"
        elif attribute == Attribute.zustand:
            return "zustandsdomaene"
        elif attribute == Attribute.geschlecht:
            return None
        elif attribute == Attribute.ortsbegebenheit:
            return None
        else:
            Logger.error("No name for attribute \"" + str(attribute) + "\" specified")
            raise ValueError("No name for attribute \"" + str(attribute) + "\" specified")
        return
