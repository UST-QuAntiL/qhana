# QHAna - Quantum Humanities Analyse Tool
The Quantum Humanities Analyse Tool is a toolset of Machine Learning techniques designed for the use with the MUSE Repository.

## Installation
It is recommended to use the code within a virtual environment like [Anaconda Distribution](https://www.anaconda.com/distribution).
All necessary packages can be installed using the *qhana.yml* file located within this repository.
It is recommended to use a linux environment.
To create the virtual environment, one can use the conda terminal and type in the following command:

`conda env create -f <path>`

with \<path\> being the path to the *qhana.yml* file.

The following packages (including their dependencies) are needed and will be installed when using the *qhana.yml* file:

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
| flask                  | https://anaconda.org/anaconda/flask                  |
| pygraphviz             | https://anaconda.org/anaconda/pygraphviz             |
| qiskit                 | https://qiskit.org                                   |
| cvxpy                  | https://anaconda.org/conda-forge/cvxpy               |
| pylatexenc             | https://anaconda.org/conda-forge/pylatexenc          |

## Access to the database
In order to get access to the database, one has to create a *config.ini* file such as:

[mysql]<br/>
host = &lt;host address or IP&gt;<br/>
user =  &lt;username&gt;<br/>
password =  &lt;password&gt;<br/>
database = KostuemRepo<br/>

The file must lay in the top directory, i.e. the same directory as the *main.py* script.

## Usage - Web GUI

In order to use the web gui, the following command needs to be executed:

```
python web.py
```

Thereby, a WSGI server will be started in the background and the home page of the application will be shown automatically.
This is the recommended approach to run the QHAna Tool.
The command Line interface do not cover all the features.

## Usage - Command Line

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
