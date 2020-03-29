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

## Access to the database
In order to get access to the database, one has to create a *config.ini* file such as:

[mysql]<br/>
host = \<host address or IP\><br/>
user = \<username\><br/>
password = \<password\><br/>
database = KostuemRepo<br/>

The file must lay in the top directory, i.e. the same directory as the *main.py* script.
