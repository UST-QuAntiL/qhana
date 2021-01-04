from abc import *
from clusteringAlgorithm import ClusteringAlgorithm
import numpy as np


class QuantumClusteringAlgorithm(ABC, ClusteringAlgorithm):
    """
    A base class for quantum algorithms that run on IBMQ.
    """

    @abstractmethod
    async def perform_clustering(self):
        pass

    def __init__(self, backend, max_qubits, shots_per_circuit):
        self.backend = backend
        self.max_qubits = max_qubits
        self.shots_per_circuit = shots_per_circuit

    @property
    def backend(self):
        return self.__backend

    @backend.setter
    def backend(self, value):
        self._backend = value

    @property
    def max_qubits(self):
        return self.__max_qubits

    @max_qubits.setter
    def max_qubits(self, value):
        self._max_qubits = value

    @property
    def shots_per_circuit(self):
        return self.__shots_per_circuit

    @shots_per_circuit.setter
    def shots_per_circuit(self, value):
        self._shots_per_circuit = value

    @staticmethod
    def _map_histogram_to_qubit_hits(histogram):
        """
        Maps the histogram (dictionary) to a 2D np.array with the format
        qubit_i = [#hits |0>, #hits |1>].
        """

        # Create array and store the hits per qubit, i.e. [#|0>, #|1>]
        length = int(len(list(histogram.keys())[0]))
        qubit_hits = np.zeros((length, 2))

        for basis_state in histogram:
            for i in range(0, length):
                if basis_state[length - i - 1] == "0":
                    qubit_hits[i][0] = qubit_hits[i][0] + histogram[basis_state]
                else:
                    qubit_hits[i][1] = qubit_hits[i][1] + histogram[basis_state]

        return qubit_hits

    @classmethod
    def _calculate_odd_qubits_1_hits(cls, histogram):
        """
        Calculates for all odd qubits how often they hit the
        |1> state. We use the format [odd_qubit_i, #hits|1>].
        """

        # the length is half the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        length = int(len(list(histogram.keys())[0]) / 2)
        hits = np.zeros(length)

        qubit_hits_map = cls._map_histogram_to_qubit_hits(histogram)
        for i in range(0, int(qubit_hits_map.shape[0] / 2)):
            hits[i] = int(qubit_hits_map[i * 2][1])

        return hits

    @classmethod
    def _calculate_odd_qubits_0_hits(cls, histogram):
        """
        Calculates for all odd qubits how often they hit the
        |0> state. We use the format [odd_qubit_i, #hits|0>].
        """

        # the length is half the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        length = int(len(list(histogram.keys())[0]) / 2)
        hits = np.zeros(length)

        qubit_hits_map = cls._map_histogram_to_qubit_hits(histogram)
        for i in range(0, int(qubit_hits_map.shape[0] / 2)):
            hits[i] = int(qubit_hits_map[i * 2][0])

        return hits
