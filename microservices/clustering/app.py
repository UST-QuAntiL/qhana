"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from quart import Quart, request, jsonify
import os
from negativeRotation import NegativeRotation
from destructiveInterference import DestructiveInterference
from sklearnClustering import SklearnClustering
import asyncio
import aiohttp
from numpySerializer import NumpySerializer
from threading import Thread
from quantumBackendFactory import QuantumBackendFactory

app = Quart(__name__)
app.config["DEBUG"] = True
loop = asyncio.get_event_loop()

if __name__ == "__main__":
    loop.run_until_complete(app.run_task())


# TODO: use conda install -c conda-forge flask-swagger-ui in environment!
# TODO: use conda install -c conda-forge flask-swagger-ui
# TODO: use conda install -c conda-forge quart
# TODO: use conda install -c conda-forge aiohttp

async def fetch_data_as_text(session, url):
    async with session.get(url) as response:
        return await response.text()


async def download_to_file(url, file_path):
    async with aiohttp.ClientSession() as session:
        content_as_text = await fetch_data_as_text(session, url)
        text_file = open(file_path, 'w')
        text_file.write(content_as_text)
        text_file.close()


def run_negative_rotation_clustering(
        data,
        output_file_path,
        max_qubits=5,
        shots_per_circuit=8192,
        k=2,
        max_runs=100,
        eps=5,
        backend_name='aer_qasm_simulator'):

    try:
        # get parameters
        backend = QuantumBackendFactory.create_backend(backend_name)

        # run the clustering
        algorithm = NegativeRotation(backend, max_qubits, shots_per_circuit, k, max_runs, eps)
        result = algorithm.perform_clustering(data)

        # serialize the output data
        NumpySerializer.serialize(result, output_file_path)

    except Exception as ex:
        file = open(output_file_path, 'w')
        file.write(str(ex))
        file.close()


def run_destructive_interference_clustering(
        data,
        output_file_path,
        max_qubits=5,
        shots_per_circuit=8192,
        k=2,
        max_runs=100,
        eps=5,
        backend_name='aer_qasm_simulator'):

    try:
        # get parameters
        backend = QuantumBackendFactory.create_backend(backend_name)

        # run the clustering
        algorithm = DestructiveInterference(backend, max_qubits, shots_per_circuit, k, max_runs, eps)
        result = algorithm.perform_clustering(data)

        # serialize the output data
        NumpySerializer.serialize(result, output_file_path)

    except Exception as ex:
        file = open(output_file_path, 'w')
        file.write(str(ex))
        file.close()


def run_sklearn_clustering(
        data,
        output_file_path,
        k=2,
        max_runs=100,
        eps=10e-4):

    try:
        # run the clustering
        algorithm = SklearnClustering(k, max_runs, eps)
        result = algorithm.perform_clustering(data)

        # serialize the output data
        NumpySerializer.serialize(result, output_file_path)

    except Exception as ex:
        file = open(output_file_path, 'w')
        file.write(str(ex))
        file.close()


@app.route('/')
async def index():
    return 'Hello World'


@app.route('/api/negative-rotation', methods=['GET'])
async def perform_negative_rotation_clustering():
    """
    Trigger the negative rotation clustering algorithm.
    We have the following parameters (name : type : default : description):
    input_data_url : string : : download location of the input data
    job_id : int : : the id of the job
    backend_name : string : aer_qasm_simulator : the name for the quantum backend
    max_qubits : int : 5 : the maximum amount of qubits used in parallel
    shots_per_circuit : 8192 : int - how many shots per circuit
    k : int : 2 : amount of clusters
    max_runs : int : 100 : how many runs for the iterative procedure
    eps : float : 5 : convergence condition
    """

    # load the data from url
    input_data_url = request.args.get('input_data_url', type=str)
    job_id = request.args.get('job_id', type=int)
    backend_name = request.args.get('backend_name', type=str, default='aer_qasm_simulator')
    max_qubits = request.args.get('max_qubits', type=int, default=5)
    shots_per_circuit = request.args.get('shots_per_circuit', type=int, default=8192)
    k = request.args.get('k', type=int, default=2)
    max_runs = request.args.get('max_runs', type=int, default=100)
    eps = request.args.get('eps', type=float, default=5.0)

    input_file_path = './static/input' + str(job_id)
    output_file_path = './static/output' + str(job_id)

    # response parameters
    message = "success"
    status_code = 200

    try:
        # delete old files if exist
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # download the input data and store it locally
        await download_to_file(input_data_url, input_file_path)

        # deserialize the input data
        data = NumpySerializer.deserialize(input_file_path)

        # perform the real clustering, i.e. start
        # a fire and forget the clustering task
        thread = Thread(target=run_negative_rotation_clustering,
                        kwargs={
                            'data': data,
                            'output_file_path': output_file_path,
                            'backend_name': backend_name,
                            'max_qubits': max_qubits,
                            'shots_per_circuit': shots_per_circuit,
                            'k': k,
                            'max_runs': max_runs,
                            'eps': eps
                        })
        thread.start()

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code)


@app.route('/api/destructive-interference', methods=['GET'])
async def perform_destructive_interference_clustering():
    """
    Trigger the destructive interference clustering algorithm.
    We have the following parameters (name : type : default : description):
    input_data_url : string : : download location of the input data
    job_id : int : : the id of the job
    backend_name : string : aer_qasm_simulator : the name for the quantum backend
    max_qubits : int : 5 : the maximum amount of qubits used in parallel
    shots_per_circuit : 8192 : int - how many shots per circuit
    k : int : 2 : amount of clusters
    max_runs : int : 100 : how many runs for the iterative procedure
    eps : float : 5 : convergence condition
    """

    # load the data from url
    input_data_url = request.args.get('input_data_url', type=str)
    job_id = request.args.get('job_id', type=int)
    backend_name = request.args.get('backend_name', type=str, default='aer_qasm_simulator')
    max_qubits = request.args.get('max_qubits', type=int, default=5)
    shots_per_circuit = request.args.get('shots_per_circuit', type=int, default=8192)
    k = request.args.get('k', type=int, default=2)
    max_runs = request.args.get('max_runs', type=int, default=100)
    eps = request.args.get('eps', type=float, default=5.0)

    input_file_path = './static/input' + str(job_id)
    output_file_path = './static/output' + str(job_id)

    # response parameters
    message = "success"
    status_code = 200

    try:
        # delete old files if exist
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # download the input data and store it locally
        await download_to_file(input_data_url, input_file_path)

        # deserialize the input data
        data = NumpySerializer.deserialize(input_file_path)

        # perform the real clustering, i.e. start
        # a fire and forget the clustering task
        thread = Thread(target=run_destructive_interference_clustering,
                        kwargs={
                            'data': data,
                            'output_file_path': output_file_path,
                            'backend_name': backend_name,
                            'max_qubits': max_qubits,
                            'shots_per_circuit': shots_per_circuit,
                            'k': k,
                            'max_runs': max_runs,
                            'eps': eps
                        })
        thread.start()

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code)


@app.route('/api/sklearn', methods=['GET'])
async def perform_sklearn_clustering():
    """
    Trigger the sklearn clustering algorithm.
    We have the following parameters (name : type : default : description):
    input_data_url : string : : download location of the input data
    job_id : int : : the id of the job
    k : int : 2 : amount of clusters
    max_runs : int : 100 : how many runs for the iterative procedure
    eps : float : 10e-4 : convergence condition frobenius norm
    """

    # load the data from url
    input_data_url = request.args.get('input_data_url', type=str)
    job_id = request.args.get('job_id', type=int)
    k = request.args.get('k', type=int, default=2)
    max_runs = request.args.get('max_runs', type=int, default=100)
    eps = request.args.get('eps', type=float, default=10e-4)

    input_file_path = './static/input' + str(job_id)
    output_file_path = './static/output' + str(job_id)

    # response parameters
    message = "success"
    status_code = 200

    try:
        # delete old files if exist
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        # download the input data and store it locally
        await download_to_file(input_data_url, input_file_path)

        # deserialize the input data
        data = NumpySerializer.deserialize(input_file_path)

        # perform the real clustering, i.e. start
        # a fire and forget the clustering task
        thread = Thread(target=run_sklearn_clustering,
                        kwargs={
                            'data': data,
                            'output_file_path': output_file_path,
                            'k': k,
                            'max_runs': max_runs,
                            'eps': eps
                        })
        thread.start()

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code)


@app.route('/api/output<int:job_id>', methods=['GET'])
def get_output(job_id):
    if request.method == 'GET':
        # define file paths
        input_file_path = './static/input' + str(job_id)
        output_file_path = './static/output' + str(job_id)

        if not os.path.exists(input_file_path):
            message = 'no job'
            status_code = 404
            output_data_url = ''
        elif not os.path.exists(output_file_path):
            message = 'job running'
            status_code = 201
            output_data_url = ''
        else:
            message = 'job finished'
            status_code = 200
            output_data_url = request.url_root[:-4] + 'static/output' + str(job_id)

        return jsonify(message=message, status_code=status_code, output_data_url=output_data_url)
