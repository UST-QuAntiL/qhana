from backend.entityService import EntityService
from backend.attributeComparer import AttributeComparerType
from backend.entity import Costume, CostumeFactory, Entity, EntityFactory
from backend.elementComparer import ElementComparerType
from backend.entityComparer import CostumeComparer, EmptyAttributeAction
from backend.transformer import TransformerType
from backend.aggregator import AggregatorType
from backend.attribute import Attribute
from backend.database import Database
from backend.logger import Logger, LogLevel
import numpy as np
from typing import List
import random as rd
from backend.entityService import Subset

# class creates and manage similarity matrizes 
class EntitySimilarities():
    """
    Initializes similarties 
    """
    def __init__(
        self,
        costume_plan: List = [AggregatorType.mean,
                              TransformerType.linearInverse,
                            (
                                Attribute.dominanteFarbe,
                                ElementComparerType.wuPalmer,
                                AttributeComparerType.symMaxMean,
                                EmptyAttributeAction.ignore
                            )],
        bool_memory: bool = False,
        entity_number: int = 2147483646,
        subsetEnum: Subset = None,
        useRandom: bool = False
    ) -> None:
        self.__costume_plan: List = costume_plan
        self.__invalid_entity_index: List[int] = []
        self.__valid_entity_index: List[int] = []
        self.__last_sequenz: List[int] = []
        self.__bool_memory: bool = bool_memory
        db = Database()
        db.open()
        self.__service: EntityService = EntityService()
        self.__service.add_plan(self.__costume_plan)

        if subsetEnum == None:
            self.__service.create_entities(db, entity_number, filter_rules=self.__service.filterRules)
            self.__entity_number: int = len(self.__service.get_entities(useRandom))
        else:
            self.__service.create_subset(subsetEnum, db)
            self.__entity_number: int = len(self.__service.get_entities(useRandom))

        self.__service.create_components()

        if self.__bool_memory:
            self.__memory = np.eye(self.__entity_number+1,self.__entity_number+1) + np.ones((self.__entity_number+1,self.__entity_number+1)) * (-1.0)
            Logger.warning("Memory in Similarity Class was activated. This needs {0} Bytes ({1} MB) more memory space.".format(round(self.__memory.size* self.__memory.itemsize,0),round(self.__memory.size/1000000* self.__memory.itemsize,2)))
        else:
            Logger.warning("Memory in Similarity Class was not activated.")
        return


    # getter methodes
    # get entity_number
    def get_entity_number(self) -> List[int]:
        return self.__entity_number
    # get costume_plan
    def get_costume_plan(self) -> List[int]:
        return self.__costume_plan
    # get invalid_entity_index
    def get_invalid_entity_index(self) -> List[int]:
        return self.__invalid_entity_index
    # get valid_entity_index
    def get_valid_entity_index(self) -> List[int]:
        return self.__valid_entity_index
    # get last_sequenz
    def get_last_sequenz(self) -> List[int]:
        return self.__last_sequenz
    # get last_sequenz_id
    def get_last_sequenz_id(self) -> List[int]:
        entities: List[Entity] = self.__service.get_entities()
        last_sequenz_id: List[int] = []
        for key in self.__last_sequenz:
            last_sequenz_id.append(entities[key].id)
        return last_sequenz_id
    # get List of Entities
    def get_list_entities(self) -> List[Entity]:
        return self.__service.get_entities()
    # get bool memory
    def get_bool_memory(self) -> bool:
        return self.__bool_memory
    # get memory
    def get_memory(self) -> np.matrix:
        if self.__bool_memory:
            return self.__memory
        else:
            Logger.warning("No Memory initialized. The Programm will be quit know.")
            return np.zeros((1, 1))
    # entities in memory
    def get_entities_in_memory(self) -> str:
        EIM: str = ""
        sequenz_bool : bool = False
        for number in range(self.__entity_number):
            if sequenz_bool == False:
                if number in self.__valid_entity_index or number in self.__invalid_entity_index:
                    EIM += str(number) + "-"
                    sequenz_bool = True
            else:
                if  not (number in self.__valid_entity_index or number in self.__invalid_entity_index):
                    EIM += str(number-1) + "; "
                    sequenz_bool = False
                elif (number in self.__valid_entity_index or number in self.__invalid_entity_index) and number == self.__entity_number-1:
                    EIM += str(number) + "; "
                    sequenz_bool = False
        if self.__bool_memory:
            return EIM
        else:
            return "No memory initialized"

                


# setter and getter params
    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        similaritiesTypeName = "Entity Similarities"
        params.append(("name", "Similarities Type" ,"description", similaritiesTypeName ,"header"))
        parameter_entity_number = self.get_entity_number()
        params.append(("entityNumber", "Entity Number" ,"description", parameter_entity_number, "header" ))
        parameter_costume_plan = self.get_costume_plan()
        params.append(("costumePlan", "Costume Plan" ,"description", parameter_costume_plan, "header"))
        parameter_entitiesInMemory_index = self.get_entities_in_memory()
        params.append(("EIM", "Entities in Memory" ,"description", parameter_entitiesInMemory_index , "header"))
        parameter_last_sequenz_id = self.get_last_sequenz_id()
        params.append(("lastSequenz" , "Last ID Sequenz" ,"description", parameter_last_sequenz_id, "header"))
        parameter_bool_memory = self.get_bool_memory()
        params.append(("memoryBool","Memory Initialized","description", parameter_bool_memory, "header"))
        return params

    





    # create limited similarity matrix
    def create_matrix_limited(self, first:int,last:int) -> np.matrix:
        # error handling for integer arguments
        try:
        # >>first<< is greater than >>last<<
            if last <= first:
                raise NameError('{0} is greater or equal than {1}.'.format(first,last))
        # out of bound 
            if last >= self.__entity_number:
                raise IndexError(' Index {0} is greater than the last Index of Entities ({1}).'.format(last,self.__entity_number-1))
            elif last < 0:
                raise IndexError(' Index {0} is smaller than zero.'.format(last))
            elif first >= self.__entity_number:
                raise IndexError(' Index {0} is greater than the last Index of Entities ({1}).'.format(first,self.__entity_number-1))
            elif first < 0:
                raise IndexError(' Index {0} is smaller than zero.'.format(first))



        except NameError as error:
            Logger.error("C:Similarities|create_matrix_limited-> " +str(error) + " The application will quit now.")
            quit()
        except IndexError as warning:
            if last >= self.__entity_number:
                last = self.__entity_number-1
            if first >= self.__entity_number:
                first = self.__entity_number-1
            if first < 0:
                first = 0
            if last < 0:
                last = 0
            Logger.warning("C:Similarities|create_matrix_limited-> " +str(warning) + " The application will handle now: first = {0} and last = {1}".format(first,last))
        
        #check weather entities already been tested
        self.__comparing_check(first,last)

        #removing invalid entities from entities_index
        entity_index: List[int] = list(range(first,last+1))
        size: int = len(entity_index)
        count: int = 0
        for i in self.__invalid_entity_index:
            if i in entity_index:
                count += 1
                entity_index.remove(i)  
             
        Logger.warning("{0} entities from {1} are not comparable. These invalid entities won't be used for the similarity matrix.".format(count,size))
        self.__last_sequenz = entity_index
        # create for valid entities the similarity_matrix
        total_number: int = len(entity_index)
        similarity_matrix: np.matrix = np.zeros((total_number,total_number))
        counter: int = 0
        total_counter: int =  (len(entity_index)-1)*len(entity_index)/2
        counter_percent: int = 20
        for i in range(len(entity_index)):
            for j in range(len(entity_index)):
                if j < i:
                    counter += 1
                    if counter % (total_counter/counter_percent) < 0.5 or counter % (total_counter/counter_percent) > total_counter/counter_percent-0.5:
                        Logger.normal(str(round(counter/total_counter*100))+ '%')
                    if not self.__bool_memory:
                        comparedResult = self.__service.calculate_distance(entity_index[i], entity_index[j])
                        Logger.debug("compared result from costume {0} with costume {1} is {2} (No Memory initialized).".format(str(entity_index[i]),str(entity_index[j]),str(round(comparedResult, 2))))
                        similarity_matrix[i,j] = comparedResult
                        similarity_matrix[j,i] = comparedResult
                    else:
                        if self.__memory[entity_index[i],entity_index[j]] < -0.5:
                            comparedResult = self.__service.calculate_distance(entity_index[i], entity_index[j])
                            Logger.debug("Not Recovered from Memory: compared result from costume {0} with costume {1} is {2} (Memory initalized).".format(str(entity_index[i]),str(entity_index[j]),str(round(comparedResult, 2))))
                            similarity_matrix[i,j] = comparedResult
                            similarity_matrix[j,i] = comparedResult
                            self.__memory[entity_index[i],entity_index[j]] = comparedResult
                            self.__memory[entity_index[j],entity_index[i]] = comparedResult
                        else:
                            comparedResult = self.__memory[entity_index[i],entity_index[j]]
                            Logger.debug("    Recovered from Memory: compared result from costume {0} with costume {1} is {2} (Memory initialized).".format(str(entity_index[i]),str(entity_index[j]),str(round(comparedResult, 2))))
                            similarity_matrix[i,j] = comparedResult
                            similarity_matrix[j,i] = comparedResult
        return similarity_matrix
    # create similarity matrix for all valid entities List
    def create_matrix_all(self) -> np.matrix:
        first : int = 0
        last : int = len(self.__entity_number)-1
        return self.create_matrix_limited(first, last)
 
    # check if entities valid for comparing (sort entities(it's index) in __valid_entities_index if valid for comparing ...
    # and in __invalid_entities_index if it's not comparable)
    def __comparing_check(self, first: int, last: int)-> None:
        #return
        # check whether area has already been tested
        entity_index: List[int] = list(range(first,last+1))
        rem_list: List[int] = []
        for i in entity_index:
            if i in self.__valid_entity_index:
                rem_list.append(i)
            elif i in self.__invalid_entity_index:
                rem_list.append(i)
        for i in rem_list:
            entity_index.remove(i)
        if len(entity_index) == 0:
            return
        # error handling for entities objects
        # find in entity_index one object which is valid
        random_first: int
        if len(self.__valid_entity_index) == 0:  
            check: bool = True
            count: int = 0
            while check:
                count += 1
                try:
                    if count == 200:
                        Logger.error("C:Similarities|create_matrix_limited-> No costume was found for checking valid entities. The application will quit now.")
                        quit()
                    random_first = rd.choice(entity_index)
                    self.__service.calculate_distance(random_first, random_first)
                    check = False
                except Exception:
                    check = True
        else:
            random_first = rd.choice(self.__valid_entity_index)
        
        # find valid entities in range(first,last)
        rem_list = []
        for i in entity_index:
            try:
                Logger.debug("costume {0} compared with costume {1}".format(random_first,i))
                self.__service.calculate_distance(random_first, i)
                self.__valid_entity_index.append(i)
            except Exception:
                Logger.warning("Costume {0} not comparable. This entry will be skipped for the similarity matrix.".format(i))
                self.__invalid_entity_index.append(i)
        return

    




