from qiskit import QuantumCircuit
from qiskit.providers.aer import Aer
from qiskit.aqua.quantum_instance import QuantumInstance

class CircuitExecutor():

    @classmethod
    def runCircuit(cls, circuit: QuantumCircuit, parameterizations: list, backend, shots, add_measurements=False):
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

        Qbackend = Aer.get_backend(backend)
        QInstance = QuantumInstance(Qbackend, seed_simulator=9283712, seed_transpiler=9283712, shots=shots)

        # add measurements
        if not QInstance.is_statevector and add_measurements:
            circuit.barrier()
            circuit.measure(circuit.qubits, circuit.clbits)

        results = []

        for parameterization in parameterizations:
            parameterization = parameterization_from_parameter_names(circuit, parameterization)
            curr_circuit = circuit.assign_parameters(parameterization)
            curr_circuit = QInstance.transpile(curr_circuit)[0]

            # run each circuit individually (for now), this can be optimized
            result = QInstance.execute(curr_circuit, had_transpiled=True)

            results.append(result)

        return results, QInstance.is_statevector

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