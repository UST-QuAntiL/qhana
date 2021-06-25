from typing import List, Callable, Dict, Tuple

import pennylane as qml

from backend.tools import pl_samples_to_counts


class CircuitExecutor:

    @classmethod
    def runCircuits(
            cls, circuit_function: Callable[[List[float], List[float]], List], input_params: List[List[float]],
            weight_params: List[List[float]], n_qubits: int, backend, token, shots, add_measurements=False)\
            -> Tuple[List[Dict[str, int]], bool]:
        """
            Runs the circuit with each parameterization in the provided list and
            on the provided quantum instance and

            parameters
                - circuit: a QuantumCircuit to run
                - parameterizations: a list of dictionaries [parameter name] -> [value]
                - add_measurements: (optional) adds measurement operations if instance is not a statevector instance

            returns
                - results: a list of the results from these runs
                - is_statevector: True if QInstance is statevector
        """

        dev = qml.device("default.qubit", wires=n_qubits, shots=1024)  # TODO: replace with selected backend
        circuit = qml.QNode(circuit_function, dev)
        results = []

        for inp, weights in zip(input_params, weight_params):
            counts = pl_samples_to_counts(circuit(inp, weights))
            results.append(counts)

        return results, False  # TODO: statevector


def parameterization_from_parameter_names(circuit, parameterization):
    """
        converts the dict [parameter name] -> [value] to
        [parameter] -> [value]
        i.e. replaces parameter names by actual parameters in circuit
    """

    parameters = circuit.parameters
    parameterization_new = {}
    for param_name in parameterization:
        for parameter in parameters:
            if parameter.name == param_name:
                parameterization_new[parameter] = parameterization[param_name]
                break
    return parameterization_new