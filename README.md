# PlanQK - Machine Learning with Taxonomies
This repository contains implementations for using different similarity measures for machine learning purposes.
The code is designed for using the MUSE dataset, which is provided by the Institute of Architecture of Application Systems (IAAS) from the University of Stuttgart.
Python 3.6 is required in order to use this implementation.

## Installation
It is recommended to use the code within a virtual environment like [Anaconda Distribution](https://www.anaconda.com/distribution).
All necessarry packages can be installed using the yml file located within this repository.
To create the virtual environment, one can use the conda terminal and type in the following command:

`conda env create -f <path>`

with `<path>` being the path to the planqk.yml file

In order to get access to the database, one has to create a config.ini file such as:

[mysql]<br/>
host = localhost<br/>
user = <br/>
password = <br/>
database = KostuemRepo<br/>
