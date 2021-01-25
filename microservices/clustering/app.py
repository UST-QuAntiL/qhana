"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from quart import Quart, request, jsonify
import os
import asyncio
import numpy as np
from numpySerializer import NumpySerializer
from qiskitSerializer import QiskitSerializer
from threading import Thread
from quantumBackendFactory import QuantumBackendFactory
from clusteringCircuitGenerator import ClusteringCircuitGenerator
from dataProcessingService import DataProcessingService
from clusteringCircuitExecutor import ClusteringCircuitExecutor
from convergenceCalculationService import ConvergenceCalculationService
from fileService import FileService

app = Quart(__name__)
app.config["DEBUG"] = True
loop = asyncio.get_event_loop()


def generate_url(url_root, route, file_name):
    return url_root + '/static/' + route + '/' + file_name + '.txt'


@app.route('/')
async def index():
    return 'QHana Clustering Microservice'


@app.route('/api/centroid-calculation/initialization/<int:job_id>', methods=['POST'])
async def initialize_centroids(job_id):
    """
    Create k random centroids in the range [0, 1] x [0, 1].
    """

    # load the data from url
    k = request.args.get('k', type=int, default=2)

    centroids_file_path = './static/centroid-calculation/initialization/centroids' \
                          + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    centroids_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/centroid-calculation/initialization/')

        # delete old files if exist
        FileService.delete_if_exist(centroids_file_path)

        # generate k centroids
        centroids = DataProcessingService.generate_random_data(k)

        # serialize the data
        NumpySerializer.serialize(centroids, centroids_file_path)

        # generate urls
        url_root = request.host_url
        centroids_url = generate_url(url_root,
                                     'centroid-calculation/initialization',
                                     'centroids' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message,
                   status_code=status_code,
                   centroids_url=centroids_url)


@app.route('/api/angle-calculation/rotational-clustering/<int:job_id>', methods=['POST'])
async def calculate_angles(job_id):
    """
    Performs the pre processing of a general rotational clustering algorithm,
    i.e. the angle calculations.

    We take the data and centroids and calculate the centroid and data angles.
    """

    # load the data from url or json body
    data_url = request.args.get('data_url', type=str)
    if data_url is None:
        data_url = (await request.get_json())['data_url']
    centroids_url = request.args.get('centroids_url', type=str)
    if centroids_url is None:
        centroids_url = (await request.get_json())['centroids_url']
    base_vector_x = request.args.get('base_vector_x', type=float, default=1.0)
    base_vector_y = request.args.get('base_vector_y', type=float, default=0.0)

    data_file_path = './static/angle-calculation/rotational-clustering/data' \
                     + str(job_id) + '.txt'
    centroids_file_path = './static/angle-calculation/rotational-clustering/centroids' \
                          + str(job_id) + '.txt'
    centroid_angles_file_path = './static/angle-calculation/rotational-clustering/centroid_angles' \
                                + str(job_id) + '.txt'
    data_angles_file_path = './static/angle-calculation/rotational-clustering/data_angles' \
                            + str(job_id) + '.txt'

    base_vector = np.array([base_vector_x, base_vector_y])

    # response parameters
    message = 'success'
    status_code = 200
    data_angles_url = ''
    centroid_angles_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/angle-calculation/rotational-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(data_file_path,
                                    centroids_file_path,
                                    centroid_angles_file_path,
                                    data_angles_file_path)

        # download the data and store it locally
        await FileService.download_to_file(data_url, data_file_path)
        await FileService.download_to_file(centroids_url, centroids_file_path)

        # deserialize the data
        data = NumpySerializer.deserialize(data_file_path)
        centroids = NumpySerializer.deserialize(centroids_file_path)

        # map data and centroids to standardized unit sphere
        data = DataProcessingService.normalize(DataProcessingService.standardize(data))
        centroids = DataProcessingService.normalize(DataProcessingService.standardize(centroids))

        # calculate the angles
        data_angles = DataProcessingService.calculate_angles(data, base_vector)
        centroid_angles = DataProcessingService.calculate_angles(centroids, base_vector)

        # serialize the data
        NumpySerializer.serialize(data_angles, data_angles_file_path)
        NumpySerializer.serialize(centroid_angles, centroid_angles_file_path)

        # generate urls
        url_root = request.host_url
        data_angles_url = generate_url(url_root,
                                       'angle-calculation/rotational-clustering',
                                       'data_angles' + str(job_id))
        centroid_angles_url = generate_url(url_root,
                                           'angle-calculation/rotational-clustering',
                                           'centroid_angles' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message,
                   status_code=status_code,
                   data_angles_url=data_angles_url,
                   centroid_angles_url=centroid_angles_url)


@app.route('/api/circuit-generation/negative-rotation-clustering/<int:job_id>', methods=['POST'])
async def generate_negative_rotation_circuits(job_id):
    """
    Generates the negative rotation clustering quantum circuits.

    We take the data and centroid angles and return a url to a file with the
    quantum circuits as qasm strings.
    """

    # load the data from url or json body
    data_angles_url = request.args.get('data_angles_url', type=str)
    if data_angles_url is None:
        data_angles_url = (await request.get_json())['data_angles_url']
    centroid_angles_url = request.args.get('centroid_angles_url', type=str)
    if centroid_angles_url is None:
        centroid_angles_url = (await request.get_json())['centroid_angles_url']
    max_qubits = request.args.get('max_qubits', type=int, default=5)

    data_angles_file_path = './static/circuit-generation/negative-rotation-clustering/data_angles' \
                            + str(job_id) + '.txt'
    centroid_angles_file_path = './static/circuit-generation/negative-rotation-clustering/centroid_angles' \
                                + str(job_id) + '.txt'
    circuits_file_path = './static/circuit-generation/negative-rotation-clustering/circuits' \
                         + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    circuits_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/circuit-generation/negative-rotation-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(data_angles_file_path, centroid_angles_file_path, circuits_file_path)

        # download the data and store it locally
        await FileService.download_to_file(data_angles_url, data_angles_file_path)
        await FileService.download_to_file(centroid_angles_url, centroid_angles_file_path)

        # deserialize the data and centroid angles
        data_angles = NumpySerializer.deserialize(data_angles_file_path)
        centroid_angles = NumpySerializer.deserialize(centroid_angles_file_path)

        # perform circuit generation
        circuits = ClusteringCircuitGenerator.generate_negative_rotation_clustering(max_qubits,
                                                                                    data_angles,
                                                                                    centroid_angles)

        # serialize the quantum circuits
        QiskitSerializer.serialize(circuits, circuits_file_path)

        # generate url
        url_root = request.host_url
        circuits_url = generate_url(url_root,
                                    'circuit-generation/negative-rotation-clustering',
                                    'circuits' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, circuits_url=circuits_url)


@app.route('/api/circuit-generation/destructive-interference-clustering/<int:job_id>', methods=['POST'])
async def generate_destructive_interference_circuits(job_id):
    """
    Generates the destructive interference clustering quantum circuits.

    We take the data and centroid angles and return a url to a file with the
    quantum circuits as qasm strings.
    """

    # load the data from url
    data_angles_url = request.args.get('data_angles_url', type=str)
    centroid_angles_url = request.args.get('centroid_angles_url', type=str)
    max_qubits = request.args.get('max_qubits', type=int, default=5)

    data_angles_file_path = './static/circuit-generation/destructive-interference-clustering/data_angles' \
                            + str(job_id) + '.txt'
    centroid_angles_file_path = './static/circuit-generation/destructive-interference-clustering/centroid_angles' \
                                + str(job_id) + '.txt'
    circuits_file_path = './static/circuit-generation/destructive-interference-clustering/circuits' \
                         + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    circuits_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/circuit-generation/destructive-interference-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(data_angles_file_path, centroid_angles_file_path, circuits_file_path)

        # download the data and store it locally
        await FileService.download_to_file(data_angles_url, data_angles_file_path)
        await FileService.download_to_file(centroid_angles_url, centroid_angles_file_path)

        # deserialize the data and centroid angles
        data_angles = NumpySerializer.deserialize(data_angles_file_path)
        centroid_angles = NumpySerializer.deserialize(centroid_angles_file_path)

        # perform circuit generation
        circuits = ClusteringCircuitGenerator.generate_destructive_interference_clustering(max_qubits,
                                                                                           data_angles,
                                                                                           centroid_angles)

        # serialize the quantum circuits
        QiskitSerializer.serialize(circuits, circuits_file_path)

        # generate url
        url_root = request.host_url
        circuits_url = generate_url(url_root,
                                    'circuit-generation/destructive-interference-clustering',
                                    'circuits' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, circuits_url=circuits_url)


@app.route('/api/circuit-execution/negative-rotation-clustering/<int:job_id>', methods=['POST'])
async def execute_negative_rotation_circuits(job_id):
    """
    Executes the negative rotation clustering algorithm given the generated
    quantum circuits.
    """

    # load the data from url
    circuits_url = request.args.get('circuits_url', type=str)
    if circuits_url is None:
        circuits_url = (await request.get_json())['circuits_url']
    k = request.args.get('k', type=int)
    if k is None:
        k = (await request.get_json())['k']
    backend_name = request.args.get('backend_name', type=str)
    if backend_name is None:
        backend_name = (await request.get_json())['backend_name']
    shots_per_circuit = request.args.get('shots_per_circuit', type=int, default=8192)

    circuits_file_path = './static/circuit-execution/negative-rotation-clustering/circuits' \
                         + str(job_id) + '.txt'
    cluster_mapping_file_path = './static/circuit-execution/negative-rotation-clustering/cluster_mapping' \
                                + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    cluster_mapping_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/circuit-execution/negative-rotation-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(circuits_file_path)

        # download the circuits and store it locally
        await FileService.download_to_file(circuits_url, circuits_file_path)

        # deserialize the circuits
        circuits = QiskitSerializer.deserialize(circuits_file_path)

        # create the quantum backend
        backend = QuantumBackendFactory.create_backend(backend_name)

        # execute the circuits
        cluster_mapping = ClusteringCircuitExecutor.execute_negative_rotation_clustering(circuits,
                                                                                         k,
                                                                                         backend,
                                                                                         shots_per_circuit)

        # serialize the data
        NumpySerializer.serialize(cluster_mapping, cluster_mapping_file_path)

        # generate urls
        url_root = request.host_url
        cluster_mapping_url = generate_url(url_root,
                                           'circuit-execution/negative-rotation-clustering',
                                           'cluster_mapping' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, cluster_mapping_url=cluster_mapping_url)


@app.route('/api/circuit-execution/destructive-interference-clustering/<int:job_id>', methods=['POST'])
async def execute_destructive_interference_circuits(job_id):
    """
    Executes the destructive interference clustering algorithm given the generated
    quantum circuits.
    """

    # load the data from url
    circuits_url = request.args.get('circuits_url', type=str)
    k = request.args.get('k', type=int)
    backend_name = request.args.get('backend_name', type=str, default='aer_qasm_simulator')
    shots_per_circuit = request.args.get('shots_per_circuit', type=int, default=8192)

    circuits_file_path = './static/circuit-execution/destructive-interference-clustering/circuits' \
                         + str(job_id) + '.txt'
    cluster_mapping_file_path = './static/circuit-execution/destructive-interference-clustering/cluster_mapping' \
                                + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    cluster_mapping_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/circuit-execution/destructive-interference-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(circuits_file_path)

        # download the circuits and store it locally
        await FileService.download_to_file(circuits_url, circuits_file_path)

        # deserialize the circuits
        circuits = QiskitSerializer.deserialize(circuits_file_path)

        # create the quantum backend
        backend = QuantumBackendFactory.create_backend(backend_name)

        # execute the circuits
        cluster_mapping = ClusteringCircuitExecutor \
            .execute_destructive_interference_clustering(circuits,
                                                         k,
                                                         backend,
                                                         shots_per_circuit)

        # serialize the data
        NumpySerializer.serialize(cluster_mapping, cluster_mapping_file_path)

        # generate urls
        url_root = request.host_url
        cluster_mapping_url = generate_url(url_root,
                                           'circuit-execution/destructive-interference-clustering',
                                           'cluster_mapping' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, cluster_mapping_url=cluster_mapping_url)


@app.route('/api/centroid-calculation/rotational-clustering/<int:job_id>', methods=['POST'])
async def calculate_centroids(job_id):
    """
    Performs the post processing of a general rotational clustering algorithm,
    i.e. the centroid calculations.

    We take the cluster mapping, data and old centroids and calculate the
    new centroids.
    """

    # load the data from url
    data_url = request.args.get('data_url', type=str)
    if data_url is None:
        data_url = (await request.get_json())['data_url']
    cluster_mapping_url = request.args.get('cluster_mapping_url', type=str)
    if cluster_mapping_url is None:
        cluster_mapping_url = (await request.get_json())['cluster_mapping_url']
    old_centroids_url = request.args.get('old_centroids_url', type=str)
    if old_centroids_url is None:
        old_centroids_url = (await request.get_json())['old_centroids_url']

    data_file_path = './static/centroid-calculation/rotational-clustering/data' \
                     + str(job_id) + '.txt'
    cluster_mapping_file_path = './static/centroid-calculation/rotational-clustering/cluster_mapping' \
                                + str(job_id) + '.txt'
    old_centroids_file_path = './static/centroid-calculation/rotational-clustering/old_centroids' \
                              + str(job_id) + '.txt'
    centroids_file_path = './static/centroid-calculation/rotational-clustering/centroids' \
                          + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    centroids_url = ''

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/centroid-calculation/rotational-clustering/')

        # delete old files if exist
        FileService.delete_if_exist(data_file_path, cluster_mapping_file_path, old_centroids_file_path,
                                    centroids_file_path)

        # download the data and store it locally
        await FileService.download_to_file(data_url, data_file_path)
        await FileService.download_to_file(cluster_mapping_url, cluster_mapping_file_path)
        await FileService.download_to_file(old_centroids_url, old_centroids_file_path)

        # deserialize the data
        data = NumpySerializer.deserialize(data_file_path)
        cluster_mapping = NumpySerializer.deserialize(cluster_mapping_file_path)
        old_centroids = NumpySerializer.deserialize(old_centroids_file_path)

        # map data and centroids to standardized unit sphere
        data = DataProcessingService.normalize(DataProcessingService.standardize(data))
        old_centroids = DataProcessingService.normalize(DataProcessingService.standardize(old_centroids))

        # calculate new centroids
        centroids = DataProcessingService.calculate_centroids(cluster_mapping, old_centroids, data)

        # serialize the data
        NumpySerializer.serialize(centroids, centroids_file_path)

        # generate urls
        url_root = request.host_url
        centroids_url = generate_url(url_root,
                                     'centroid-calculation/rotational-clustering',
                                     'centroids' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, centroids_url=centroids_url)


@app.route('/api/convergence-check/<int:job_id>', methods=['POST'])
async def check_convergence(job_id):
    """
    Performs the convergence check for a general KMeans clustering algorithm.

    We take the old and new centroids, calculate their pairwise distance and sum them up
    and divide it by k.

    If the resulting value is less then the given eps, we return convergence, if not,
    we return not converged.
    """

    # load the data from url
    new_centroids_url = request.args.get('new_centroids_url', type=str)
    old_centroids_url = request.args.get('old_centroids_url', type=str)
    eps = request.args.get('eps', type=float, default=0.0001)

    old_centroids_file_path = './static/convergence-check/old_centroids' + str(job_id) + '.txt'
    new_centroids_file_path = './static/convergence-check/new_centroids' + str(job_id) + '.txt'

    # response parameters
    message = 'success'
    status_code = 200
    convergence = False
    distance = 0.0

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/convergence-check/')

        # delete old files if exist
        FileService.delete_if_exist(old_centroids_file_path, new_centroids_file_path)

        # download the data and store it locally
        await FileService.download_to_file(old_centroids_url, old_centroids_file_path)
        await FileService.download_to_file(new_centroids_url, new_centroids_file_path)

        # deserialize the data
        old_centroids = NumpySerializer.deserialize(old_centroids_file_path)
        new_centroids = NumpySerializer.deserialize(new_centroids_file_path)

        # check convergence
        distance = ConvergenceCalculationService.calculate_averaged_euclidean_distance(old_centroids, new_centroids)

        convergence = distance < eps

    except Exception as ex:
        message = str(ex)
        status_code = 500

    return jsonify(message=message, status_code=status_code, convergence=convergence, distance=distance)


@app.route('/api/circuit-generation/negative-rotation-clustering/<int:job_id>', methods=['GET'])
def get_negative_rotation_circuits(job_id):
    """
    Get the result for the negative rotation clustering quantum circuit generation.
    """

    if request.method == 'GET':
        # define file paths
        data_file_path = './static/negative_rotation_data' + str(job_id) + '.txt'
        circuits_file_path = './static/negative_rotation_circuits' + str(job_id) + '.txt'

        if not os.path.exists(data_file_path):
            message = 'no job'
            status_code = 404
            circuits_url = ''
        elif not os.path.exists(circuits_file_path):
            message = 'job running'
            status_code = 201
            circuits_url = ''
        else:
            message = 'job finished'
            status_code = 200
            circuits_url = request.url_root[:-4] + 'static/negative_rotation_circuits' + str(job_id) + '.txt'

        return jsonify(message=message, status_code=status_code, circuits_url=circuits_url)


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
        algorithm = NegativeRotationClustering(backend, max_qubits, shots_per_circuit, k, max_runs, eps)
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
        algorithm = DestructiveInterferenceClustering(backend, max_qubits, shots_per_circuit, k, max_runs, eps)
        result = algorithm.perform_clustering(data)

        # serialize the output data
        NumpySerializer.serialize(result, output_file_path)

    except Exception as ex:
        file = open(output_file_path, 'w')
        file.write(str(ex))
        file.close()


def run_state_preparation_clustering(
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
        algorithm = StatePreparationClustering(backend, max_qubits, shots_per_circuit, k, max_runs, eps)
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


@app.route('/api/negative-rotation-clustering', methods=['GET'])
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


@app.route('/api/destructive-interference-clustering', methods=['GET'])
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


@app.route('/api/state-preparation-clustering', methods=['GET'])
async def perform_state_preparation_clustering():
    """
    Trigger the state preparation clustering algorithm.
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
        thread = Thread(target=run_state_preparation_clustering,
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


@app.route('/api/sklearn-clustering', methods=['GET'])
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


if __name__ == "__main__":
    loop.run_until_complete(app.run_task())