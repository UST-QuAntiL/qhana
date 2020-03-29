from backend.costume import Costume
from backend.database import Database
from backend.costumeComparer import CostumeComparer
from backend.logger import Logger, LogLevel
import numpy as np
from typing import List
import random as rd

# class creates and manage similarity matrizes 
class Similarities():
    """
    Initializes similarties 
    """
    def __init__(
        self,
        costumeComparer: CostumeComparer = CostumeComparer(),
        costumes: List[Costume] = [],
        bool_memory: bool = False
    ) -> None:
        self.__costumeComparer = costumeComparer
        self.__invalid_costumes_index: List[int] = []
        self.__valid_costumes_index: List[int] = []
        self.__last_sequenz: List[int] = []
        self.__bool_memory = bool_memory
        if len(costumes) == 0:
            db = Database()
            db.open()
            self.__costumes = db.get_costumes()
        else:
            self.__costumes = costumes
        if self.__bool_memory:
            self.__memory = np.eye(len(self.__costumes),len(self.__costumes)) + np.ones((len(self.__costumes),len(self.__costumes))) * (-1.0)
            Logger.warning("Memory in Similarity Class was activated. This needs {0} Bytes ({1} MB) more memory space.".format(round(self.__memory.size* self.__memory.itemsize,0),round(self.__memory.size/1000000* self.__memory.itemsize,2)))
        else:
            Logger.warning("Memory in Similarity Class was not activated.")
        return
    @classmethod
    def only_costumes(self, costumes: List[Costume]= [], bool_memory: bool = False):
        costumeComparer: CostumeComparer = CostumeComparer()
        return Similarities(costumeComparer,costumes, bool_memory)
    @classmethod
    def only_costumeComparer(self, costumeComparer: CostumeComparer = CostumeComparer(),bool_memory: bool = False):
        costumes: List[Costume] = []
        return Similarities(costumeComparer,costumes,bool_memory)

    # getter methodes
    # get invalid_costumes_index
    def get_invalid_costumes_index(self) -> List[int]:
        return self.__invalid_costumes_index
    # get valid_costumes_index
    def get_valid_costumes_index(self) -> List[int]:
        return self.__valid_costumes_index
    # get last_sequenz
    def get_last_sequenz(self) -> List[int]:
        return self.__last_sequenz
    # get List of Costumes
    def get_list_costumes(self) -> List[Costume]:
        return self.__costumes
    # get bool memory
    def get_bool_memory(self) -> bool:
        return self.__bool_memory
    # get memory
    def get_memory(self) -> np.matrix:
        if self.__bool_memory:
            return self.__memory
        else:
            Logger.error("No Memory initialized. The Programm will be quit know.")
            quit()

    # create limited similarity matrix
    def create_matrix_limited(self, first:int,last:int) -> np.matrix:
        costumes: List[Costume] = self.__costumes
        # error handling for integer arguments
        try:
        # >>first<< is greater than >>last<<
            if last <= first:
                raise NameError('{0} is greater or equal than {1}.'.format(first,last))
        # out of bound 
            if last >= len(costumes):
                raise IndexError(' Index {0} is greater than the last Index of Costumes ({1}).'.format(last,len(costumes)-1))
            elif last < 0:
                raise IndexError(' Index {0} is smaller than zero.'.format(last))
            elif first >= len(costumes):
                raise IndexError(' Index {0} is greater than the last Index of Costumes ({1}).'.format(first,len(costumes)-1))
            elif first < 0:
                raise IndexError(' Index {0} is smaller than zero.'.format(first))



        except NameError as error:
            Logger.error("C:Similarities|create_matrix_limited-> " +str(error) + " The application will quit now.")
            quit()
        except IndexError as warning:
            if last >= len(costumes):
                last = len(costumes)-1
            if first >= len(costumes):
                first = len(costumes)-1
            if first < 0:
                first = 0
            if last < 0:
                last = 0
            Logger.warning("C:Similarities|create_matrix_limited-> " +str(warning) + " The application will handle now: first = {0} and last = {1}".format(first,last))
        
        #check weather costumes already been tested
        self.__comparing_check(first,last)

        #removing invalid costumes from costumes_index
        costumes_index: List[int] = list(range(first,last+1))
        size: int = len(costumes_index)
        count: int = 0
        for i in self.__invalid_costumes_index:
            if i in costumes_index:
                count += 1
                costumes_index.remove(i)  
             
        Logger.warning("{0} costumes from {1} are not comparable. These invalid costumes won't be used for the similarity matrix.".format(count,size))
        self.__last_sequenz = costumes_index
        # create for valid costumes the similarity_matrix
        total_number: int = len(costumes_index)
        similarity_matrix: np.matrix = np.zeros((total_number,total_number))
        for i in range(len(costumes_index)):
            for j in range(len(costumes_index)):
                if j < i:
                    if not self.__bool_memory:
                        comparedResult = self.__costumeComparer.compare_distance(costumes[costumes_index[i]], costumes[costumes_index[j]])
                        Logger.debug("compared result from costume {0} with costume {1} is {2} (No Memory initialized).".format(str(costumes_index[i]),str(costumes_index[j]),str(round(comparedResult, 2))))
                        similarity_matrix[i,j] = comparedResult
                        similarity_matrix[j,i] = comparedResult
                    else:
                        if self.__memory[costumes_index[i],costumes_index[j]] < -0.5:
                            comparedResult = self.__costumeComparer.compare_distance(costumes[costumes_index[i]], costumes[costumes_index[j]])
                            Logger.debug("Not Recovered from Memory: compared result from costume {0} with costume {1} is {2} (Memory initalized).".format(str(costumes_index[i]),str(costumes_index[j]),str(round(comparedResult, 2))))
                            similarity_matrix[i,j] = comparedResult
                            similarity_matrix[j,i] = comparedResult
                            self.__memory[costumes_index[i],costumes_index[j]] = comparedResult
                            self.__memory[costumes_index[j],costumes_index[i]] = comparedResult
                        else:
                            comparedResult = self.__memory[costumes_index[i],costumes_index[j]]
                            Logger.debug("    Recovered from Memory: compared result from costume {0} with costume {1} is {2} (Memory initialized).".format(str(costumes_index[i]),str(costumes_index[j]),str(round(comparedResult, 2))))
                            similarity_matrix[i,j] = comparedResult
                            similarity_matrix[j,i] = comparedResult
        return similarity_matrix
    # create similarity matrix for all valid costumes List
    def create_matrix_all(self) -> np.matrix:
        first : int = 0
        last : int = len(self.__costumes)-1
        return self.create_matrix_limited(first, last)
 
    # check if costumes valid for comparing (sort costumes(it's index) in __valid_costumes_index if valid for comparing ...
    # and in __invalid_costumes_index if it's not comparable)
    def __comparing_check(self, first: int, last: int)-> None:
        costumes: List[Costume] = self.__costumes
        # check whether area has already been tested
        costumes_index: List[int] = list(range(first,last+1))
        rem_list: List[int] = []
        for i in costumes_index:
            if i in self.__valid_costumes_index:
                rem_list.append(i)
            elif i in self.__invalid_costumes_index:
                rem_list.append(i)
        for i in rem_list:
            costumes_index.remove(i)
        if len(costumes_index) == 0:
            return
        # error handling for costumes objects
        # find in costumes_index one object which is valid
        random_first: int
        if len(self.__valid_costumes_index) == 0:  
            check: bool = True
            count: int = 0
            while check:
                count += 1
                try:
                    if count == 200:
                        Logger.error("C:Similarities|create_matrix_limited-> No costume was found for checking valid costumes. The application will quit now.")
                        quit()
                    random_first = rd.choice(costumes_index)
                    self.__costumeComparer.compare_distance(costumes[random_first], costumes[random_first])
                    check = False
                except Exception:
                    check = True
        else:
            random_first = rd.choice(self.__valid_costumes_index)
        
        # find valid costumes in range(first,last)
        rem_list = []
        for i in costumes_index:
            try:
                Logger.debug("costume {0} compared with costume {1}".format(random_first,i))
                self.__costumeComparer.compare_distance(costumes[random_first], costumes[i])
                self.__valid_costumes_index.append(i)
            except Exception:
                Logger.warning("Costume {0} not comparable. This entry will be skipped for the similarity matrix.".format(i))
                self.__invalid_costumes_index.append(i)
        return

    




