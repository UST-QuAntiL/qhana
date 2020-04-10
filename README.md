# PlanQK - Machine Learning with Taxonomies
This repository contains implementations for using different similarity measures for machine learning purposes.
The code is designed for using the MUSE dataset, which is provided by the Institute of Architecture of Application Systems (IAAS) from the University of Stuttgart.
Python 3.6 is required in order to use this implementation.

## Installation
It is recommended to use the code within a virtual environment like [Anaconda Distribution](https://www.anaconda.com/distribution).
All necessarry packages can be installed using the *planqk.yml* file located within this repository.
It is recommended to use a linux environment.
To create the virtual environment, one can use the conda terminal and type in the following command:

`conda env create -f <path>`

with \<path\> being the path to the *planqk.yml* file.

This software library is fully compatible with Windows Subsystem for Linux.

The following packages (including their dependencies) are needed and will be installed when using the planqk.yml file:

| Name                   | Source                                               |
| ---------------------- | ---------------------------------------------------- |
| mysql-connector-python | https://anaconda.org/anaconda/mysql-connector-python |
| networkx               | https://anaconda.org/anaconda/networkx               |
| numpy                  | https://anaconda.org/anaconda/numpy                  |
| matplotlib             | https://anaconda.org/conda-forge/matplotlib          |
| simplejson             | https://anaconda.org/anaconda/simplejson             |
| colorama               | https://anaconda.org/anaconda/colorama               |
| scikit-learn           | https://anaconda.org/anaconda/scikit-learn           |
| pandas                 | https://anaconda.org/anaconda/pandas                 |

## Access to the database
In order to get access to the database, one has to create a *config.ini* file such as:

[mysql]<br/>
host = \<host address or IP\><br/>
user = \<username\><br/>
password = \<password\><br/>
database = KostuemRepo<br/>

The file must lay in the top directory, i.e. the same directory as the *main.py* script.

## Usage

The program is devided into commands.
A program call has always the following structure:

```
main.py <global arguments> <command> <command arguments>
```

To create all the available taxonomies into a folder called "tax" one can run the following
```
main.py -ll 3 create_taxonomies -o tax
```

### Global Arguments

We have the following global arguments:

`-ll LOG_LEVEL, --log_level LOG_LEVEL` - log level for the current session: 0 - nothing, 1 - errors [default], 2 - warnings, 3 - debug

`-db DATABASE_CONFIG_FILE, --database_config_file DATABASE_CONFIG_FILE` - filepath for the *.ini file for the database connection

### Commands

At the moment, the following commands are availabel:

#### Create Taxonomies

`create_taxonomies` - creates the taxonomies from the database (svg, json and dot)

##### Arguments

` -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY` - specifies the directory for the output [default: /taxonomies]

#### List implemented Taxonomies

`list_implemented_taxonomies` - lists all the implemented taxonomies that can be used for machine learning

#### List implemented Attributes

`list_implemented_attributes` - lists all the implemented attributes that can be used for machine learning

#### List implemented Attribute Comparer

`list_implemented_attribute_comparer` - lists all the implemented attribute comparer that can be used for machine learning

#### List implemented Aggregator

`list_implemented_aggregator` - lists all the implemented aggregator that can be used for machine learning

#### List implemented Element Comparer

`list_implemented_element_comparer` - lists all the implemented element comparer that can be used for machine learning

#### List implemented Transformer

`list_implemented_transformer` - lists all the implemented transformer that can be used for machine learning
