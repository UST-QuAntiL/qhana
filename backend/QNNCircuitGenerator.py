# circuit adapted from https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
from typing import List, Callable

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parameter import Parameter
from numpy import pi
from qiskit.circuit.classicalregister import ClassicalRegister
import pennylane as qml


class QNNCircuitGenerator():
    @classmethod
    def genCircuit(cls, n_qbits: int, depth: int = 4, measure: bool = False) -> Callable[[List[float], List[float]], List]:
        # qr = QuantumRegister(n_qbits, 'q')
        # cr = ClassicalRegister(n_qbits, 'c')
        # qc = QuantumCircuit(qr, cr)

        def circ_func(input_params: List[float], weights: List[float]) -> List:
            """
                Embedding layer
            """

            # qc.h(qr)
            for i in range(n_qbits):
                qml.Hadamard(wires=i)

            # input_params = [Parameter("in"+str(i)) for i in range(n_qbits)]

            for i in range(n_qbits):
                # qc.ry(input_params[i]*pi/2, qr[i])
                qml.RY(input_params[i] * pi / 2, wires=i)

            # qc.barrier()

            """
                Trainable Circuit
            """
            # weights = [Parameter("weight"+str(i)) for i in range(n_qbits*depth)]
            for i in range(0, n_qbits*depth, n_qbits):
                # entangle odd and even qbits among each other
                for j in range(0, n_qbits - 1, 2):
                    # qc.cnot(qr[j], qr[j+1])
                    qml.CNOT(wires=[j, j + 1])

                for j in range(1, n_qbits - 1, 2):
                    # qc.cnot(qr[j], qr[j+1])
                    qml.CNOT(wires=[j, j + 1])

                # rotate each qbit according to weight
                for j in range(n_qbits):
                    # qc.ry(weights[i+j], qr[j])
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
