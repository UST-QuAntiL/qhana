"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

import enum


class ClusteringType(enum.Enum):
    """
    Enum of all possible clustering types. Values starting with 1 are
    classical algorithms and values starting with 2 are quantum
    algorithms, i.e. classical 1X = classical, 2X = quantum.
    """

    Optics = 11
    ClassicalNaiveMaxCut = 12
    SdpMaxCut = 13
    BmMaxCut = 14
    ClassicalKMeans = 15

    QuantumMaxCut = 21
    NegativeRotationQuantumKMeans = 22
    DestructiveInterferenceQuantumKMeans = 23
    StatePreparationQuantumKMeans = 24
