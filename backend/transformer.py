from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
import math

""" 
Defines an enum to list up all available transformer
"""
class TransformerType(enum.Enum):
    linearInverse = 1
    exponentialInverse = 2
    gaussianInverese = 3
    polynomialInverse = 4

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
        elif type == TransformerType.gaussianInverese:
            return GaussianInverseTransformer()
        elif type == TransformerType.polynomialInverse:
            return PolynomialInverseTransformer()
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
    Defines the parameters of the polynomial inverse transformer
    """
    alpha = 1.0
    beta = 1.0
    """ 
    Returns the 1 / (1 + (s/alpha)^beta) as distance with s beeing the similarity
    """
    def transform(self, similarity: float) -> float:
        return 1.0 / (1.0 + pow(similarity / alpha, beta))
