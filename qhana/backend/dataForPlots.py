import numpy as np
from qhana.backend.entity import Costume
from typing import List
from sklearn.decomposition import PCA


class DataForPlots():
    def __init__(
        self,
        similarity_matrix: np.matrix = None,
        sequenz: List[int] = None,
        costumes: List[Costume] = None,
        position_matrix: np.matrix = None,
        labels: np.matrix = None,
        decision_fun = None,
        support_vectors: np.matrix = None
    ) -> None:
        self.__similarity_matrix: np.matrix = similarity_matrix
        self.__sequenz: List[int] = sequenz
        self.__position_matrix_orig: np.matrix = position_matrix
        self.__labels: np.matrix = labels
        self.__costumes: List[Costume] = costumes
        self.__decision_fun = decision_fun
        self.__support_vectors: np.matrix = support_vectors

        pca = PCA(n_components=2)
        self.__position_matrix: np.matrix = pca.fit_transform(position_matrix) # 2 dimensional
        self.__transform2d = pca.transform
        self.__inverse_transform = pca.inverse_transform

    # getter methodes
    def get_similarity_matrix(self) -> np.matrix:
        return self.__similarity_matrix
    def get_sequenz(self) -> List[int]:
        return self.__sequenz
    def get_position_matrix(self) -> np.matrix:
        return self.__position_matrix
    def get_position_matrix_orig(self) -> np.matrix:
        return self.__position_matrix_orig
    def get_labels(self) -> np.matrix:
        return self.__labels
    def get_list_costumes(self) -> List[Costume]:
        return self.__costumes
    def get_decision_fun(self):
        return self.__decision_fun
    def get_transform2d(self):
        return self.__transform2d
    def get_inverse_transform(self):
        return self.__inverse_transform
    def get_support_vectors(self) -> np.matrix:
        return self.__support_vectors
