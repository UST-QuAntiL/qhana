# circuit adapted from https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
from typing import List, Callable

from numpy import pi
import pennylane as qml


class QNNCircuitGenerator():
    @classmethod
    def genCircuit(cls, n_qbits: int, depth: int = 4, measure: bool = False) -> Callable[[List[float], List[float]], List]:
        def circ_func(input_params: List[float], weights: List[float]) -> List:
            """
                Embedding layer
            """
            for i in range(n_qbits):
                qml.Hadamard(wires=i)

            for i in range(n_qbits):
                qml.RY(input_params[i] * pi / 2, wires=i)

            """
                Trainable Circuit
            """
            for i in range(0, n_qbits*depth, n_qbits):
                # entangle odd and even qbits among each other
                for j in range(0, n_qbits - 1, 2):
                    qml.CNOT(wires=[j, j + 1])

                for j in range(1, n_qbits - 1, 2):
                    qml.CNOT(wires=[j, j + 1])

                # rotate each qbit according to weight
                for j in range(n_qbits):
                    qml.RY(weights[i + j], wires=j)

            """
                Measurement layer
                - optional to allow for statevector simulators
            """
            # if measure:
                # qc.barrier()
                # qc.measure(qr, cr)
            return [qml.sample(qml.PauliZ(wires=i)) for i in range(n_qbits)]
            # else:
            #     pass  # TODO: statevector

        return circ_func
