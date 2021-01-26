The clustering.yml file can be used to create a conda environment:

conda env create -f clustering.yml


The requirements.txt file can be used to install the dependencies when using pip:

pip install -r requirements.txt


The clustering microservice is using Quart as hosting library and Hypercorn as ASGI server.
The RestAPI can be run with

hypercorn app:app

when being in the working directory of the clustering microservice.
