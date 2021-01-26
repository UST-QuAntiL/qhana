from quart import Quart, request, jsonify
import asyncio
from fileService import FileService
from numpySerializer import NumpySerializer
from variationalSVMCircuitGenerator import VariationalSVMCircuitGenerator
from qiskitSerializer import QiskitSerializer
from SPSAOptimizer import SPSAOptimizer
import sys

app = Quart(__name__)
app.config["DEBUG"] = True
loop = asyncio.get_event_loop()


def generate_url(url_root, route, file_name):
    return url_root + '/static/' + route + '/' + file_name + '.txt'


@app.route('/')
async def index():
    return 'QHana Classification Microservice'

@app.route('/api/variational-svm-classification/initialization/<int:job_id>', methods=['POST'])
async def initialize_classification(job_id):
    """
        Initialize variational SVM classification
        * generates circuit template
        * initializes optimization parameters
    """
    # load the data from url or json body
    data_url = request.args.get('data-url', type=str)
    if data_url is None:
        data_url = (await request.get_json())['data-url']

    optimizer_parameters_url = request.args.get('optimizer-parameters-url', type=str)
    if optimizer_parameters_url is None:
        optimizer_parameters_url = (await request.get_json())['optimizer-parameters-url']

    entanglement = request.args.get('entanglement', type=str, default='full')
    feature_map_reps = request.args.get('feature-map-reps', type=int, default=1)
    variational_form_reps = request.args.get('variational-form-reps', type=int, default=3)

    # response parameters
    message = 'success'
    status_code = 200

    # file paths (inputs)
    data_file_path = './static/variational-svm-classification/initialization/data' \
                     + str(job_id) + '.txt'
    optimizer_parameters_file_path = './static/variational-svm-classification/initialization/optimizer-parameters' \
                     + str(job_id) + '.txt'

    # file paths (outputs)
    circuit_template_file_path = './static/variational-svm-classification/initialization/circuit-template' \
                         + str(job_id) + '.txt'
    thetas_file_path = './static/variational-svm-classification/initialization/thetas' \
                         + str(job_id) + '.txt'
    thetas_plus_file_path = './static/variational-svm-classification/initialization/thetas-plus' \
                         + str(job_id) + '.txt'
    thetas_minus_file_path = './static/variational-svm-classification/initialization/thetas-minus' \
                         + str(job_id) + '.txt'
    delta_file_path = './static/variational-svm-classification/initialization/delta' \
                         + str(job_id) + '.txt'
    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/variational-svm-classification/initialization/')

        # delete old files if exist
        FileService.delete_if_exist(data_file_path,
                                    optimizer_parameters_file_path,
                                    circuit_template_file_path,
                                    thetas_file_path,
                                    thetas_plus_file_path,
                                    thetas_minus_file_path,
                                    delta_file_path)

        # download the data and store it locally
        await FileService.download_to_file(data_url, data_file_path)
        await FileService.download_to_file(optimizer_parameters_url, optimizer_parameters_file_path)

        # deserialize the data
        data = NumpySerializer.deserialize(data_file_path)

        # TODO: Make feature map and variational form selectable
        # generate circuit template
        n_dimensions = data.shape[1]
        circuit_template, feature_map_parameters, var_form_parameters = \
                VariationalSVMCircuitGenerator.generateCircuitTemplate(n_dimensions, feature_map_reps, variational_form_reps, entanglement)

        # store circuit template and parameter lists
        QiskitSerializer.serialize([circuit_template], circuit_template_file_path)

        # TODO: Optimizer initialization is still specific to SPSA optimizer -> generalize
        # deserialize optimizer parameters
        optimizer_parameters = NumpySerializer.deserialize(optimizer_parameters_file_path)
        if len(optimizer_parameters) is not 5:
            raise Exception("Wrong number of optimizer parameters. 5 parameters c0 through c4 expected.")

        # initialize thetas for optimization
        n_thetas = len(var_form_parameters)
        thetas, thetas_plus, thetas_minus, delta = SPSAOptimizer.initializeOptimization(n_thetas, optimizer_parameters)

        NumpySerializer.serialize(thetas, thetas_file_path)
        NumpySerializer.serialize(thetas_plus, thetas_plus_file_path)
        NumpySerializer.serialize(thetas_minus, thetas_minus_file_path)
        NumpySerializer.serialize(delta, delta_file_path)

        # generate urls
        url_root = request.host_url
        circuit_template_url = generate_url(url_root,
                                  'variational-svm-classification/initialization',
                                  'circuit-template' + str(job_id))
        thetas_url = generate_url(url_root,
                                  'variational-svm-classification/initialization',
                                  'thetas' + str(job_id))
        thetas_plus_url = generate_url(url_root,
                                  'variational-svm-classification/initialization',
                                  'thetas-plus' + str(job_id))
        thetas_minus_url = generate_url(url_root,
                                  'variational-svm-classification/initialization',
                                  'thetas-minus' + str(job_id))
        delta_url = generate_url(url_root,
                                  'variational-svm-classification/initialization',
                                  'delta' + str(job_id))


    except Exception as ex:
        message = str(ex)
        status_code = 500
        return jsonify(message=message, status_code=status_code)

    return jsonify(message=message,
                   status_code=status_code,
                   circuit_template_url=circuit_template_url,
                   thetas_url=thetas_url,
                   thetas_plus_url=thetas_plus_url,
                   thetas_minus_url=thetas_minus_url,
                   delta_url=delta_url)

@app.route('/api/variational-svm-classification/parameterization-generation/<int:job_id>', methods=['POST'])
async def generate_circuit_parameterizations(job_id):
    """
        Generate circuit parameterizations
        * takes circuit template, data, and thetas to generate parameterizations for the circuit execution
    """

    # response parameters
    message = 'success'
    status_code = 200

    # load the data from url or json body
    data_url = request.args.get('data-url', type=str)
    if data_url is None:
        data_url = (await request.get_json())['data-url']

    circuit_template_url = request.args.get('circuit_template_url', type=str)
    if circuit_template_url is None:
        circuit_template_url = (await request.get_json())['circuit_template_url']

    thetas_url = request.args.get('thetas_url', type=str)
    if thetas_url is None:
        thetas_url = (await request.get_json())['thetas_url']

    thetas_plus_url = request.args.get('thetas_plus_url', type=str)
    if thetas_plus_url is None:
        thetas_plus_url = (await request.get_json())['thetas_plus_url']

    thetas_minus_url = request.args.get('thetas_minus_url', type=str)
    if thetas_minus_url is None:
        thetas_minus_url = (await request.get_json())['thetas_minus_url']


    # file paths (inputs)
    data_file_path = './static/variational-svm-classification/circuit-generation/data' \
                     + str(job_id) + '.txt'
    circuit_template_file_path = './static/variational-svm-classification/circuit-generation/circuit-template' \
                     + str(job_id) + '.txt'
    thetas_file_path = './static/variational-svm-classification/circuit-generation/thetas' \
                     + str(job_id) + '.txt'
    thetas_plus_file_path = './static/variational-svm-classification/circuit-generation/thetas-plus' \
                     + str(job_id) + '.txt'
    thetas_minus_file_path = './static/variational-svm-classification/circuit-generation/thetas-minus' \
                     + str(job_id) + '.txt'

    # file paths (outputs)
    parameterizations_file_path = './static/variational-svm-classification/circuit-generation/parameterizations' \
                     + str(job_id) + '.txt'

    try:
        # create working folder if not exist
        FileService.create_folder_if_not_exist('./static/variational-svm-classification/circuit-generation/')

        # download and store locally
        await FileService.download_to_file(data_url, data_file_path)
        await FileService.download_to_file(circuit_template_url, circuit_template_file_path)
        await FileService.download_to_file(thetas_url, thetas_file_path)
        await FileService.download_to_file(thetas_plus_url, thetas_plus_file_path)
        await FileService.download_to_file(thetas_minus_url, thetas_minus_file_path)

        # deserialize inputs
        data = NumpySerializer.deserialize(data_file_path)
        # TODO: Fix problems with deserialization
        circuit = QiskitSerializer.deserialize(circuit_template_file_path)
        thetas = NumpySerializer.deserialize(thetas_file_path)
        thetas_plus = NumpySerializer.deserialize(thetas_plus_file_path)
        thetas_minus = NumpySerializer.deserialize(thetas_minus_file_path)

        # generate parameterizations
        parameterizations = VariationalSVMCircuitGenerator.generateCircuitParameterizations(circuit, data, [thetas, thetas_plus, thetas_minus])

        # serialize outputs
        NumpySerializer.serialize(parameterizations, parameterizations_file_path)

        url_root = request.host_url
        parameterizations_url = generate_url(url_root,
                                  'variational-svm-classification/circuit-generation',
                                  'parameterizations' + str(job_id))

    except Exception as ex:
        message = str(ex)
        status_code = 500
        return jsonify(message=message, status_code=status_code)

    return jsonify(message=message,
                   status_code=status_code,
                   parameterizations_url=parameterizations_url)

@app.route('/api/variational-svm-classification/circuit-execution/<int:job_id>', methods=['POST'])
async def execute_circuits(job_id):
    pass

@app.route('/api/variational-svm-classification/optimization/<int:job_id>', methods=['POST'])
async def optimize(job_id):
    pass

if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except Exception as ex:
        print("Usage: {} <port>".format(sys.argv[0]))
        exit()
    loop.run_until_complete(app.run_task(port=port))