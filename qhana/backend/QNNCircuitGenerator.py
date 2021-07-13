# circuit adapted from https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parameter import Parameter
from numpy import pi
from qiskit.circuit.classicalregister import ClassicalRegister

class QNNCircuitGenerator():
    @classmethod
    def genCircuit(cls, n_qbits, depth=4, measure=False):
        qr = QuantumRegister(n_qbits, 'q')
        cr = ClassicalRegister(n_qbits, 'c')
        qc = QuantumCircuit(qr, cr)

        """
            Embedding layer
        """

        qc.h(qr)
        input_params = [Parameter("in"+str(i)) for i in range(n_qbits)]
        for i in range(n_qbits):
            qc.ry(input_params[i]*pi/2, qr[i])
        qc.barrier()

        """
            Trainable Circuit
        """

        weights = [Parameter("weight"+str(i)) for i in range(n_qbits*depth)]
        for i in range(0, n_qbits*depth, n_qbits):
            # entangle odd and even qbits among each other
            for j in range(0, n_qbits - 1, 2):
                qc.cnot(qr[j], qr[j+1])
            for j in range(1, n_qbits - 1, 2):
                qc.cnot(qr[j], qr[j+1])
            # rotate each qbit according to weight
            for j in range(n_qbits):
                qc.ry(weights[i+j], qr[j])

        """
            Measurement layer
            - optional to allow for statevector simulators
        """
        if (measure):
            qc.barrier()
            qc.measure(qr, cr)

        return qc
