The classification.yml file can be used to create a conda environment:

conda env create -f classification.yml


The requirements.txt file can be used to install the dependencies when using pip:

pip install -r requirements.txt


The classification microservice is using Quart as hosting library and Hypercorn as ASGI server.
The RestAPI can be run with

hypercorn -b 127.0.0.1:<port> app:app

or

python app.py <port>

from the working directory of the classification microservice.
