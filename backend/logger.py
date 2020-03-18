from typing import Any
from colorama import init, Fore, Back, Style
from enum import IntEnum
import datetime
import os

"""
Represents the different log levels for the logger.

Log levels are:
0 - Nothing
1 - Errors [default]
2 - Warnings
3 - Debug
"""
class LogLevel(IntEnum):
    Nothing = 0,
    Errors = 1,
    Warnings = 2,
    Debug = 3

"""
Represents a logger for logging to the console and file.

Log is always produced to the console.
When specifying log_to_file = True, which is default, the output will also
be logged into a local file called yyyy_mm_dd_hh_ss_planqk.log laying in 
the log directory of theworking directory.
"""
class Logger():
    current_date_time = datetime.datetime.today()
    level = LogLevel.Errors
    log_to_file = True
    directory = "log"
    file_name = current_date_time.strftime("%Y_%m_%d_%H_%M_%S") + "_planqk"
    file_extension = "log"
    file_path = directory + "/" + file_name + "." + file_extension
    errors_fore = Fore.RED
    errors_back = Back.BLACK
    warnings_fore = Fore.YELLOW
    warnings_back = Back.BLACK
    debug_fore = Fore.WHITE
    debug_back = Back.BLACK

    @staticmethod
    def initialize(level: LogLevel = LogLevel.Errors) -> None:
        init()
        Logger.level = level
        return

    @staticmethod
    def create_directory() -> None:
        if os.path.isdir(Logger.directory) == False:
            os.mkdir(Logger.directory)
        return

    @staticmethod
    def log(message: str, level: LogLevel) -> None:
        current_date_time = datetime.datetime.today()
        output_header = "[" + current_date_time.strftime("%Y/%m/%d-%H:%M:%S") + "]"
        fore = None
        back = None

        if level == LogLevel.Errors:
            output_header += "[ !!! ]: "
            fore = Logger.errors_fore
            back = Logger.errors_back
        elif level == LogLevel.Warnings:
            output_header += "[  !  ]: "
            fore = Logger.warnings_fore
            back = Logger.warnings_back
        else:
            output_header += "[     ]: "
            fore = Logger.debug_fore
            back = Logger.debug_back
        
        log_message = output_header + message

        print(fore + back + log_message + Style.RESET_ALL)

        if Logger.log_to_file:
            Logger.create_directory()
            with open(Logger.file_path, "a+") as file_object:
                # Move read cursor to the start of file.
                file_object.seek(0)
                # If file is not empty then append '\n'
                data = file_object.read(100)
                if len(data) > 0:
                    file_object.write("\n")
                # Append text at the end of file
                file_object.write(log_message)
        
        return

    @staticmethod
    def error(message: str) -> None:
        if int(Logger.level) >= 1:
            Logger.log(message, LogLevel.Errors)
        return
    
    @staticmethod
    def warning(message: str) -> None:
        if int(Logger.level) >= 2:
            Logger.log(message, LogLevel.Warnings)
        return
    
    @staticmethod
    def debug(message: str) -> None:
        if int(Logger.level) >= 3:
            Logger.log(message, LogLevel.Debug)
        return
