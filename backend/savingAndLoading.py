from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
from backend.dataForSavingAndLoading import DataForSavingAndLoading, DataForSimilarityMatrix, DataForScaling, DataForClustering
from backend.logger import Logger, LogLevel
import numpy as np
from typing import List
import os
import pickle
import sys as sys
import backend.scaling as scal
import backend.clustering as clu
import backend.similarities as sim

class SavingAndLoadingType(enum.Enum):
    database            = 1
    taxonomie           = 2
    similarity_matrix   = 3
    scaling             = 4
    clustering          = 5

class SavingAndLoading(metaclass=ABCMeta):
    """
    Interface for saving Object
    """
    @abstractmethod
    def saving(self) -> bool:
        pass
    @abstractmethod
    def loading(self) -> DataForSavingAndLoading:
        pass

class SavingAndLoadingFactory:
    """
    Represents the factory to create an scaling object
    """
    @staticmethod
    def create(type: SavingAndLoadingType) -> SavingAndLoading:
        if type == SavingAndLoadingType.database:
            return SAF_Database()
        if type == SavingAndLoadingType.taxonomie:
            return SAF_Taxonomie()
        if type == SavingAndLoadingType.similarity_matrix:
            return SAF_Similarity_Matrix()
        if type == SavingAndLoadingType.scaling:
            return SAF_Scaling()
        if type == SavingAndLoadingType.clustering:
            return SAF_Clustering()
        else:
            Logger.error("Unknown type of savingAndLoading. The application will quit know.")
            raise Exception("Unknown type of savingAndLoading.")


class SAF_Database(SavingAndLoading):
    def saving(self) -> bool:
        pass
    def loading(self) -> DataForSavingAndLoading:
        pass

class SAF_Taxonomie(SavingAndLoading):
    def saving(self) -> bool:
        pass
    def loading(self) -> DataForSavingAndLoading:
        pass

class SAF_Similarity_Matrix(SavingAndLoading):
    def __init__(self):
        self.type_name: SavingAndLoadingType = SavingAndLoadingType.similarity_matrix
        self.bool_set: bool = False

    def set(self,file_folder: str, similarities_object: sim.Similarities) -> bool:
        try:
            self.DataForSimilarityMatrix: DataForSimilarityMatrix = DataForSimilarityMatrix(similarities_object)
            self.file_folder: str = file_folder
            self.bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.type_name,self.file_folder, self.DataForSimilarityMatrix)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForSimilarityMatrix
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForSimilarityMatrix()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.type_name,self.file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForSimilarityMatrix()

class SAF_Scaling(SavingAndLoading):
    def __init__(self):
        self.type_name: SavingAndLoadingType = SavingAndLoadingType.scaling
        self.bool_set: bool = False

    def set(self,file_folder: str, scaling_object: scal.Scaling) -> bool:
        try:
            self.DataForScaling: DataForScaling = DataForScaling(scaling_object)
            self.file_folder: str = file_folder
            self.bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.type_name,self.file_folder, self.DataForScaling)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForScaling
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForScaling()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.type_name,self.file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForScaling()

class SAF_Clustering(SavingAndLoading):
    def __init__(self):
        self.type_name: SavingAndLoadingType = SavingAndLoadingType.clustering
        self.bool_set: bool = False

    def set(self,file_folder: str, cluster_object: clu.Clustering) -> bool:
        try:
            self.DataForClustering: DataForClustering = DataForClustering(cluster_object)
            self.file_folder: str = file_folder
            self.bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.type_name,self.file_folder, self.DataForClustering)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForClustering
        if self.bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForClustering()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.type_name,self.file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForClustering()

class _SavingAndLoadingManaging():
    
    @staticmethod
    def saving( type_name: SavingAndLoadingType, file_folder: str , save_object : DataForSavingAndLoading) -> bool:
        root_directory = "Session"
        answer: bool = True
        file_name: str = ""
        file_folder_path: str = ""
        file_folder_path = root_directory + "/" + file_folder
        
        # checking folder
        if os.path.isdir(root_directory) == False:
            os.mkdir(root_directory)
        
        if os.path.isdir(file_folder_path) == False:
            os.mkdir(file_folder_path)
        
        # file name
        if type_name == SavingAndLoadingType.database:
            file_name: str = "database"
        elif type_name == SavingAndLoadingType.taxonomie:
            file_name: str = "taxonomie"
        elif type_name == SavingAndLoadingType.similarity_matrix:
            file_name: str = "similarity_matrix"
        elif type_name == SavingAndLoadingType.scaling:
            file_name: str = "scaling"
        elif type_name == SavingAndLoadingType.clustering:
            file_name: str = "clustering"
        elif type_name == SavingAndLoadingType.plots:
            file_name: str = "plots"
        else: 
            Logger.error("No right type_name was found")
            raise Exception("No right type_name was found")

        # build file_path
        file_path = file_folder_path + "/" + file_name
        
        # check if file exists
        inputs = _Inputs()
        answer = inputs.check_file_exists(file_path, file_folder , file_name)
        
        # saving file
        try:
            if answer: 
                with open( file_path + ".pkl", 'wb') as output:
                    # pickle is not save. Trust not pickle formats generell.
                    pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
                    return True
        except Exception as error:
            Logger.warning("Saving failed: " + str(error))
            return False


    @staticmethod
    def loading( type_name: SavingAndLoadingType, file_folder: str) -> DataForSavingAndLoading:
        root_directory = "Session"
        file_name: str = ""
        file_folder_path: str = ""
        file_folder_path = root_directory + "/" + file_folder
        
        # checking folder
        if os.path.isdir(root_directory) == False:
            Logger.error("No Directory with name {0} was found.".format(root_directory))
            raise Exception("No Directory was found.")
        
        if os.path.isdir(file_folder_path) == False:
            Logger.error("No Directory with name {0} in folder {1} was found".format(file_folder,root_directory))
            raise Exception("No Directory was found.")

        # file name
        if type_name == SavingAndLoadingType.database:
            file_name: str = "database"
        elif type_name == SavingAndLoadingType.taxonomie:
            file_name: str = "taxonomie"
        elif type_name == SavingAndLoadingType.similarity_matrix:
            file_name: str = "similarity_matrix"
        elif type_name == SavingAndLoadingType.scaling:
            file_name: str = "scaling"
        elif type_name == SavingAndLoadingType.clustering:
            file_name: str = "clustering"
        elif type_name == SavingAndLoadingType.plots:
            file_name: str = "plots"
        else: 
            Logger.error("No right type_name was found")
            raise Exception("No right type_name was found")

        # build file_path
        file_path = file_folder_path + "/" + file_name


        # loading file 
        try:
            with open(file_path + ".pkl", 'rb') as input:
                return pickle.load(input)
        except Exception as error:
            raise Exception(str(error))

            


class _Inputs():
    __input: str = ""
    
    def check_file_exists(self, file_path: str , file_folder: str , file_name: str) -> bool:
        if os.path.isfile(file_path + ".pkl"):
            while True:
                self.__input = input("File with the name \"{0}\" in Session \"{1}\" already exists. Do you want to overwrite it? ([y]es/[n]o): ".format(file_name + ".pkl",file_folder))
                if self.__input == 'y':
                    return True
                elif self.__input == 'n':
                    return False
                else: 
                    Logger.warning("Wrong input. Please try again!")

        else: 
            return True