"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from rotationalKMeansClusteringAlgorithm import RotationalKMeansClusteringAlgorithm
from quantumAlgorithm import QuantumAlgorithm
from qiskit import *
import numpy as np


class DestructiveInterferenceClustering(RotationalKMeansClusteringAlgorithm, QuantumAlgorithm):
    """
    The implementation for the destructive interference quantum KMeans algorithm
    described in https://arxiv.org/abs/1909.12183.
    """

    def __init__(self, backend, max_qubits, shots_per_circuit, k, max_runs, eps, base_vector=np.array([1, 0])):
        QuantumAlgorithm.__init__(self, backend, max_qubits, shots_per_circuit)
        RotationalKMeansClusteringAlgorithm.__init__(self, k, max_runs, eps, base_vector)

    def _perform_rotational_clustering(self, centroid_angles, data_angles):
        """
        Perform the negative rotation clustering circuit.

        We return a list with a mapping from test angle indices to centroid angle indices,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        ...

        We need for each centroid-data pair two qubits.
        We do this in a chained fashion, i.e. we take the first
        data angle and use centroid_angles.shape[0] qubits.
        If we still have free qubits, we take the next test angle
        and do the same. If we reach the max_qubits limit, we execute the
        circuit and safe the result. If we are not able to test all centroids
        in one run, we just use the next run for the remaining centroids.
        """

        # this is also the amount of qubits that are needed in total
        # note that this is not necessarily in parallel, also sequential
        # is possible here
        global_work_amount = centroid_angles.shape[0] * data_angles.shape[0]

        # we store the distances as [(t1,c1), (t1,c2), ..., (t1,cn), (t2,c1), ..., (tm,cn)]
        # while each (ti,cj) stands for one distance, i.e. (ti,cj) = distance data point i
        # to centroid j
        distances = np.zeros(global_work_amount)

        # create tuples of parameters corresponding for each qubit,
        # i.e. create [t1,c1, t1,c2, ..., t1,cn, t2,c1, ..., tm,cn]
        # now with ti = data_angle_i and cj = centroid_angle_j
        parameters = []
        for i in range(0, data_angles.shape[0]):
            for j in range(0, centroid_angles.shape[0]):
                parameters.append((data_angles[i], centroid_angles[j]))

        # this is the index to iterate over all parameter pairs in the queue (parameters list)
        index = 0
        queue_not_empty = True
        circuits_index = 0
        amount_executed_circuits = 0

        # create the circuit(s)
        while queue_not_empty:
            max_qubits_for_circuit = (global_work_amount - index) * 2

            if self.max_qubits < max_qubits_for_circuit:
                qubits_for_circuit = self.max_qubits
            else:
                qubits_for_circuit = max_qubits_for_circuit

            if qubits_for_circuit % 2 != 0:
                qubits_for_circuit -= 1

            qc = QuantumCircuit(qubits_for_circuit, qubits_for_circuit)

            for i in range(0, qubits_for_circuit, 2):
                qc.h(i)
                qc.cx(i, i+1)

                relative_angular = abs(parameters[index][0] - parameters[index][1])

                # relative angular difference rotation
                qc.ry(-relative_angular, i+1)

                qc.cx(i, i+1)

                # relative angular difference rotation
                qc.ry(relative_angular, i+1)

                qc.h(i)
                qc.measure(i, i)
                qc.measure(i+1, i+1)

                index += 1
                if index == global_work_amount:
                    queue_not_empty = False
                    break

            circuits_index += 1

            # execute on IBMQ backend
            job = execute(qc, self.backend, shots=self.shots_per_circuit)

            # store the result for this sub circuit run
            amount_executed_circuits += 1
            histogram = job.result().get_counts()
            hits = self.calculate_even_qubits_1_hits(histogram)

            # the amount of hits for the |1> state is proportional
            # to the distance. Using 1 data point and one centroid,
            # i.e. 2 qubits, P|11> + P|10> is proportional to the
            # distance (but not normed)
            for i in range(0, hits.shape[0]):
                distances[index - int(qubits_for_circuit / 2) + i] = hits[i]

        return self._calculate_centroid_mapping(data_angles.shape[0], centroid_angles.shape[0], distances)
