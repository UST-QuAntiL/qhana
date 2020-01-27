from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum

""" 
Defines an enum to list up all available transformer
"""
class TransformerType(enum.Enum):
    linearInverse = 1

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

    @staticmethod
    def create(type: TransformerType) -> TransformerType:
        if type == TransformerType.linearInverse:
            return LinearInverseTransformer()
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
