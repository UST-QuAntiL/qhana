import numpy as np
from backend.costume import Costume
from typing import List

class DataForPlots():
    def __init__(
        self,
        similarity_matrix: np.matrix = None,
        sequenz: List[int] = None,
        costumes: List[Costume] = None,
        position_matrix: np.matrix = None,
        labels: np.matrix = None
    ) -> None:
        self.__similarity_matrix: np.matrix = similarity_matrix
        self.__sequenz: List[int] = sequenz
        self.__position_matrix: np.matrix = position_matrix
        self.__labels: np.matrix = labels
        self.__costumes: List[Costume] = costumes

    # getter methodes
    def get_similarity_matrix(self) -> np.matrix:
        return self.__similarity_matrix
    def get_sequenz(self) -> List[int]:
        return self.__sequenz
    def get_position_matrix(self) -> np.matrix:
        return self.__position_matrix
    def get_labels(self) -> np.matrix:
        return self.__labels
    def get_list_costumes(self) -> List[Costume]:
        return self.__costumes