from abc import ABCMeta
from abc import abstractmethod
from typing import Any
import enum
from backend.dataForSavingAndLoading import DataForSavingAndLoading, DataForCostumeSimilarities, DataForScaling, DataForClustering, DataForEntitySimilarities, DataForCostumePlan
from backend.logger import Logger, LogLevel
import numpy as np
from typing import List
import os
import pickle
import sys as sys
import backend.scaling as scal
import backend.clustering as clu
import backend.similarities as sim
import backend.entitySimilarities as esim

class SavingAndLoadingType(enum.Enum):
    costumePlan         = 1
    entitySimilarities  = 2
    costumeSimilarities = 3
    scaling             = 4
    clustering          = 5
    plots               = 6

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
        if type == SavingAndLoadingType.costumePlan:
            return SAF_Costume_Plan()
        if type == SavingAndLoadingType.entitySimilarities:
            return SAF_Entity_Similarities()
        if type == SavingAndLoadingType.costumeSimilarities:
            return SAF_Costume_Similarities()
        if type == SavingAndLoadingType.scaling:
            return SAF_Scaling()
        if type == SavingAndLoadingType.clustering:
            return SAF_Clustering()
        else:
            Logger.error("Unknown type of savingAndLoading. The application will quit know.")
            raise Exception("Unknown type of savingAndLoading.")


class SAF_Costume_Plan(SavingAndLoading):
    def __init__(self):
        self.__type_name: SavingAndLoadingType = SavingAndLoadingType.costumePlan
        self.__bool_set: bool = False

    def set(self,file_folder: str, similarities_object: List) -> bool:
        try:
            self.__DataForCostumePlan: DataForCostumePlan = DataForCostumePlan(similarities_object)
            self.__file_folder: str = file_folder
            self.__bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.__bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.__type_name,self.__file_folder, self.__DataForCostumePlan)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForCostumePlan
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForCostumePlan()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.__type_name,self.__file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForCostumePlan()

class SAF_Entity_Similarities(SavingAndLoading):
    def __init__(self):
        self.__type_name: SavingAndLoadingType = SavingAndLoadingType.entitySimilarities
        self.__bool_set: bool = False

    def set(self,file_folder: str, similarities_object: esim.EntitySimilarities) -> bool:
        try:
            self.__DataForEntitySimilarities: DataForEntitySimilarities = DataForEntitySimilarities(similarities_object)
            self.__file_folder: str = file_folder
            self.__bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.__bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.__type_name,self.__file_folder, self.__DataForEntitySimilarities)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForEntitySimilarities
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForEntitySimilarities()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.__type_name,self.__file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForEntitySimilarities()

class SAF_Costume_Similarities(SavingAndLoading):
    def __init__(self):
        self.__type_name: SavingAndLoadingType = SavingAndLoadingType.costumeSimilarities
        self.__bool_set: bool = False

    def set(self,file_folder: str, similarities_object: sim.Similarities) -> bool:
        try:
            self.__DataForCostumeSimilarities: DataForCostumeSimilarities = DataForCostumeSimilarities(similarities_object)
            self.__file_folder: str = file_folder
            self.__bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.__bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.__type_name,self.__file_folder, self.__DataForCostumeSimilarities)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForCostumeSimilarities
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForCostumeSimilarities()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.__type_name,self.__file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForCostumeSimilarities()

class SAF_Scaling(SavingAndLoading):
    def __init__(self):
        self.__type_name: SavingAndLoadingType = SavingAndLoadingType.scaling
        self.__bool_set: bool = False

    def set(self,file_folder: str, scaling_object: scal.Scaling) -> bool:
        try:
            self.__DataForScaling: DataForScaling = DataForScaling(scaling_object)
            self.__file_folder: str = file_folder
            self.__bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.__bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.__type_name,self.__file_folder, self.__DataForScaling)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForScaling
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForScaling()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.__type_name,self.__file_folder)
            return loading_object
        except Exception as error:
            Logger.warning("Loading failed. The returned object is empty now. Warning:" + str(error))
            return DataForScaling()

class SAF_Clustering(SavingAndLoading):
    def __init__(self):
        self.__type_name: SavingAndLoadingType = SavingAndLoadingType.clustering
        self.__bool_set: bool = False

    def set(self,file_folder: str, cluster_object: clu.Clustering) -> bool:
        try:
            self.__DataForClustering: DataForClustering = DataForClustering(cluster_object)
            self.__file_folder: str = file_folder
            self.__bool_set = True
        except Exception as error:
            Logger.error("set methode failed: {0}".format(str(error)))
            self.__bool_set = False
            raise Exception(str(error))

    def saving(self) -> bool:
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return False

        _SavingAndLoadingManaging.saving(self.__type_name,self.__file_folder, self.__DataForClustering)
        return True

    def loading(self) -> DataForSavingAndLoading:
        loading_object: DataForClustering
        if self.__bool_set == False:
            Logger.warning("No File was saved. The set method must be called beforehand.")
            return DataForClustering()
        try:
            loading_object = _SavingAndLoadingManaging.loading(self.__type_name,self.__file_folder)
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
        if type_name == SavingAndLoadingType.costumePlan:
            file_name: str = "costumeplan"
        elif type_name == SavingAndLoadingType.entitySimilarities:
            file_name: str = "entity_similarities"
        elif type_name == SavingAndLoadingType.costumeSimilarities:
            file_name: str = "costume_similarities"
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
        if type_name == SavingAndLoadingType.costumePlan:
            file_name: str = "costumeplan"
        elif type_name == SavingAndLoadingType.entitySimilarities:
            file_name: str = "entity_similarities"
        elif type_name == SavingAndLoadingType.costumeSimilarities:
            file_name: str = "costume_similarities"
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
                #self.__input = input("File with the name \"{0}\" in Session \"{1}\" already exists. Do you want to overwrite it? ([y]es/[n]o): ".format(file_name + ".pkl",file_folder))
                self.__input = 'y'
                if self.__input == 'y':
                    return True
                elif self.__input == 'n':
                    return False
                else: 
                    Logger.warning("Wrong input. Please try again!")

        else: 
            return True