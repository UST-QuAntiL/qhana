import time 
from backend.logger import Logger, LogLevel
import datetime

class Timer():
    __start_time: float = time.time()
    __start_bool: bool = False
    __stopp_time: float = time.time()
    __total_time: int = 0

    def start(self) -> None:
        Logger.normal("Timer has started...")
        self.__start_time = time.time()
        self.__start_bool = True

    def stop(self) -> None:
        if self.__start_bool:
            self.__stopp_time = time.time()
            self.__total_time = round(self.__stopp_time - self.__start_time)
            str_time: str = datetime.timedelta(seconds=self.__total_time)
            Logger.normal("Timer was stopped. Total time was {0} h:mm:ss".format(str_time))
        else:
            Logger.error("Timer was not started. The application will quit now.")
    
    def sleep(self, seconds: int = 0, minutes: int = 0 , hours: int = 0) -> None:
        total_seconds: int = seconds + minutes * 60 + hours * 60 * 3600
        time.sleep(total_seconds)
