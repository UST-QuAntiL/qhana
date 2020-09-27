from mysql.connector import MySQLConnection
from mysql.connector import Error
from configparser import ConfigParser
from backend.singleton import Singleton
from backend.logger import Logger, LogLevel

"""
Respresents the database class for db connection.
"""
class Database(Singleton):
    """
    Specifies the default for the config file
    """
    config_file_default = "config.ini"

    """
    Initializes the database singleton.
    """
    def __init__(self) -> None:
        self.__connection = None
        self.__cursor = None
        return
    
    """
    Deletes the database singleton.
    """
    def __del__(self) -> None:
        self.close()
        return

    """
    Opens the database using the config.ini file.
    """
    def open(self, filename = None) -> None:
        # if already connected do nothing
        if self.__connection is not None and self.__connection.is_connected():
            return

        if filename == None:
            filename = Database.config_file_default

        section = "mysql"
        parser = ConfigParser()
        parser.read(filename)
        connection_string = {}

        if parser.has_section(section):
            items = parser.items(section)
            for item in items:
                connection_string[item[0]] = item[1]
        else:
            Logger.error("{0} not found in the {1} file".format(section, filename))
            raise Exception("{0} not found in the {1} file".format(section, filename))

        self.databaseName = parser.get(section, "database")
        self.user = parser.get(section, "user")
        self.host = parser.get(section, "host")
    
        self.__connection = MySQLConnection(**connection_string)
    
        if self.__connection.is_connected():
            Logger.debug("Successfully connected to database")

    """
    Closes the database connection.
    """
    def close(self) -> None:
        if self.__connection is not None and self.__connection.is_connected():
            self.__connection.close()
            self.__cursor = None
            Logger.debug("Successfully disconnected from database")

    """
    Returns a cursor to the database.
    """
    def get_cursor(self):
        return self.__connection.cursor()
