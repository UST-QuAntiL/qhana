"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from enum import Enum


class Attribute(Enum):
    """
    An enum specifying the available attributes of the
    muse repository.
    """

    Location = "Location",
    DominantColor = "DominantColor",
    StereotypeRelevant = "StereotypeRelevant",
    DominantFunction = "DominantFunction",
    DominantCondition = "DominantCondition",
    DominantCharacterTrait = "DominantCharacterTrait",
    Stereotype = "Stereotype",
    Gender = "Gender",
    DominantAgeImpression = "DominantAgeImpression",
    Genre = "Genre",
    Profession = "Profession",
    RoleRelevance = "RoleRelevance",
    TimeOfSetting = "TimeOfSetting",
    TimeOfDay = "TimeOfDay",
    BodyModification = "BodyModification",
    MaritalStatus = "MaritalStatus",
    CharacterTrait = "CharacterTrait",
    Venue = "Venue",
    VenueDetail = "VenueDetail",
    AgeImpression = "AgeImpression",
    BaseElement = "BaseElement",
    Design = "Design",
    Form = "Form",
    WayOfWearing = "WayOfWearing",
    Condition = "Condition",
    Function = "Function",
    Material = "Material",
    MaterialImpression = "MaterialImpression",
    Color = "Color",
    ColorImpression = "ColorImpression",
    ColorConcept = "ColorConcept"
