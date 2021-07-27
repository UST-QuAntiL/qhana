"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from abc import ABCMeta
from abc import abstractmethod
import enum
import math


class TransformerType(enum.Enum):
    """
    Defines an enum to list up all available transformers.
    """

    linearInverse = 0
    exponentialInverse = 1
    gaussianInverse = 2
    polynomialInverse = 3
    squareInverse = 4


class TransformerFactory:
    """
    Represents the factory to create transformers.
    """

    @classmethod
    def create(cls, transformer_type):
        """
        Creates a transformer given the type.
        """
        if transformer_type == TransformerType.linearInverse:
            return LinearInverseTransformer()
        elif transformer_type == TransformerType.exponentialInverse:
            return ExponentialInverseTransformer()
        elif transformer_type == TransformerType.gaussianInverse:
            return GaussianInverseTransformer()
        elif transformer_type == TransformerType.polynomialInverse:
            return PolynomialInverseTransformer()
        elif transformer_type == TransformerType.squareInverse:
            return SquareInverseTransformer()
        else:
            raise Exception("Unknown type of transformer")


class Transformer(metaclass=ABCMeta):
    """
    Represents the abstract transformer base class.
    """

    @abstractmethod
    def transform(self, similarity):
        """
        Returns the distance value to the given similarity value.
        """
        pass


class LinearInverseTransformer(Transformer):
    def transform(self, similarity):
        """
        Returns the 1 - s as distance with s being the similarity
        """
        return 1.0 - similarity


class ExponentialInverseTransformer(Transformer):
    def transform(self, similarity):
        """
        Returns the exp(-s) as distance with s being the similarity
        """
        return math.exp(-similarity)


class GaussianInverseTransformer(Transformer):
    def transform(self, similarity):
        """
        Returns the exp(-s^2) as distance with s being the similarity
        """
        return math.exp(-similarity * similarity)


class PolynomialInverseTransformer(Transformer):
    def transform(self, similarity):
        """
        Returns the 1 / (1 + (s/alpha)^beta) as distance with s being the similarity
        """
        alpha = 1.0
        beta = 1.0
        return 1.0 / (1.0 + pow(similarity / alpha, beta))


class SquareInverseTransformer(Transformer):
    def transform(self, similarity):
        """
        Returns the (1/sqrt(2)) * sqrt(s(i,i) + s(j,j) - 2s(i,j)) = sqrt(2) * sqrt(2 (1 - s))
        as distance with s being the similarity and 1 as the maxSimilarity.
        Due to the factor sqrt(2) the distance is between 0 and 1.
        """
        max_similarity = 1.0
        return (1.0 / math.sqrt(2.0)) * math.sqrt(2 * max_similarity - 2 * similarity)
