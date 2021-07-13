from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import math
from backend.logger import Logger

""" 
Defines an enum to list up all available transformer
"""
class TransformerType(enum.Enum):
    linearInverse = "linearInverse"
    exponentialInverse = "exponentialInverse"
    gaussianInverse = "gaussianInverse"
    polynomialInverse = "polynomialInverse"
    squareInverse = "squareInverse"

    @staticmethod
    def get_name(transformerType) -> str:
        name = ""
        if transformerType == TransformerType.linearInverse:
            name = "LinearInverse"
        elif transformerType == TransformerType.exponentialInverse:
            name = "ExponentialInverse"
        elif transformerType == TransformerType.gaussianInverse:
            name = "GaussianInverse"
        elif transformerType == TransformerType.polynomialInverse:
            name = "PolynomialInverse"
        elif transformerType == TransformerType.squareInverse:
            name = "SquareInverse"
        else:
            Logger.error("No name for transformer \"" + str(transformerType) + "\" specified")
            raise ValueError("No name for transformer \"" + str(transformerType) + "\" specified")
        return name

    @staticmethod
    def get_description(transformerType) -> str:
        description = ""
        if transformerType == TransformerType.linearInverse:
            description += "Transforms similarities into distances using " \
                + "the linear inverse function dist(sim) = 1 - sim"
        elif transformerType == TransformerType.exponentialInverse:
            description += "Transforms similarities into distances using " \
                + "the exponential inverse function dist(sim) = exp(- sim)"
        elif transformerType == TransformerType.gaussianInverse:
            description += "Transforms similarities into distances using " \
                + "the gaussian inverse function dist(sim) = exp(- sim^2)"
        elif transformerType == TransformerType.polynomialInverse:
            description += "Transforms similarities into distances using " \
                + "the polynomial inverse function dist(sim) = 1 / (1 + (sim/alpha)^beta)"
        elif transformerType == TransformerType.squareInverse:
            description += "Transforms similarities into distances using " \
                + "the square inverse function dist(sim) = (1/sqrt(2)) * sqrt(2 * maxSim - 2 * s) " \
                + "with maxSim beeing 1.0 and the normalization factor (1/sqrt(2))."
        else:
            Logger.error("No description for transformer \"" + str(transformerType) + "\" specified")
            raise ValueError("No description for transformer \"" + str(transformerType) + "\" specified")
        return description

""" 
Represents the abstract transformer base class
"""
class Transformer(metaclass=ABCMeta):
    """ 
    Returns the distance value to the given similarity value
    """
    @abstractmethod
    def transform(self, similarity: float) -> float:
        pass

""" 
Represents the factory to create a transformer
"""
class TransformerFactory:
    """
    Creates a transformer.
    """
    @staticmethod
    def create(type: TransformerType) -> TransformerType:
        if type == TransformerType.linearInverse:
            return LinearInverseTransformer()
        elif type == TransformerType.exponentialInverse:
            return ExponentialInverseTransformer()
        elif type == TransformerType.gaussianInverse:
            return GaussianInverseTransformer()
        elif type == TransformerType.polynomialInverse:
            return PolynomialInverseTransformer()
        elif type == TransformerType.squareInverse:
            return SquareInverseTransformer()
        else:
            raise Exception("Unknown type of transformer")

"""
Represents the linear inverse transformer
"""
class LinearInverseTransformer(Transformer):
    """ 
    Returns the 1 - s as distance with s beeing the similarity
    """
    def transform(self, similarity: float) -> float:
        return 1.0 - similarity

"""
Represents the exponential inverse transformer
"""
class ExponentialInverseTransformer(Transformer):
    """ 
    Returns the exp(-s) as distance with s beeing the similarity
    """
    def transform(self, similarity: float) -> float:
        return math.exp(-similarity)

"""
Represents the gaussian inverse transformer
"""
class GaussianInverseTransformer(Transformer):
    """ 
    Returns the exp(-s^2) as distance with s beeing the similarity
    """
    def transform(self, similarity: float) -> float:
        return math.exp(-similarity * similarity)

"""
Represents the polynomial inverse transformer
"""
class PolynomialInverseTransformer(Transformer):
    """ 
    Returns the 1 / (1 + (s/alpha)^beta) as distance with s beeing the similarity
    """
    def transform(self, similarity: float) -> float:
        alpha = 1.0
        beta = 1.0
        return 1.0 / (1.0 + pow(similarity / alpha, beta))

"""
Represents the square inverse transformer
"""
class SquareInverseTransformer(Transformer):
    """ 
    Returns the (1/sqrt(2)) * sqrt(s(i,i) + s(j,j) - 2s(i,j)) = sqrt(2) * sqrt(2 (1 - s)) as distance with s beeing the similarity
    and 1 as the maxSimilarity. Due to the factor sqrt(2) the distnace is between 0 and 1.
    """
    def transform(self, similarity: float) -> float:
        maxSimilarity = 1.0
        return (1.0 / math.sqrt(2.0)) * math.sqrt(2 * maxSimilarity - 2 * similarity)
