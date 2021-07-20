from mysql.connector import MySQLConnection
from configparser import ConfigParser
from qhana.backend.singleton import Singleton
from qhana.backend.logger import Logger
import os


class Database(Singleton):
    """
    Represents the database class for db connection.
    """

    config_file_default = "config.ini"
    """
    Specifies the default for the config file
    """

    def __init__(self) -> None:
        """
        Initializes the database singleton.
        """
        self.__connection = None
        self.__cursor = None
        self.connected = False
        return

    def __del__(self) -> None:
        """
        Deletes the database singleton.
        """
        self.close()
        return

    def open(self, filename=None) -> None:
        """
        Opens the database using the config file.
        """

        # if already connected do nothing
        if self.__connection is not None and self.__connection.is_connected():
            return

        if filename is None:
            filename = Database.config_file_default

        if not os.path.exists(filename):
            Logger.error("Couldn't find config file for database connection.")

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
            self.connected = True
            Logger.debug("Successfully connected to database")

    def open_with_params(self, host: str, user: str, password: str, database: str):
        self.__connection = MySQLConnection(host=host, user=user, password=password, database=database)

        if self.__connection.is_connected():
            self.connected = True
            Logger.debug("Successfully connected to database")

    def close(self) -> None:
        """
        Closes the database connection.
        """
        if self.__connection is not None and self.__connection.is_connected():
            self.__connection.close()
            self.__cursor = None
            self.connected = False
            Logger.debug("Successfully disconnected from database")

    def get_cursor(self):
        """
        Returns a cursor to the database.
        """
        return self.__connection.cursor()
