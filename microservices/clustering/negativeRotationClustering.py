"""
Author: Daniel Fink
Email: daniel-fink@outlook.com
"""

from rotationalKMeansClusteringAlgorithm import RotationalKMeansClusteringAlgorithm
from quantumAlgorithm import QuantumAlgorithm
from qiskit import *
import numpy as np


class NegativeRotationClustering(RotationalKMeansClusteringAlgorithm, QuantumAlgorithm):
    """
    The implementation for the negative rotation quantum KMeans algorithm
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

        We need for each data angle centroid_angles.shape[0] qubits.
        We do this in a chained fashion, i.e. we take the first
        data angle and use centroid_angles.shape[0] qubits.
        If we still have free qubits, we take the next data angle
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
        # i.e. create [[t1,c1], [t1,c2], ..., [t1,cn], [t2,c1], ..., [tm,cn]]
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
            max_qubits_for_circuit = global_work_amount - index

            if self.max_qubits < max_qubits_for_circuit:
                qubits_for_circuit = self.max_qubits
            else:
                qubits_for_circuit = max_qubits_for_circuit

            qc = QuantumCircuit(qubits_for_circuit, qubits_for_circuit)

            for i in range(0, qubits_for_circuit):
                # test_angle rotation
                qc.ry(parameters[index][0], i)

                # negative centroid_angle rotation
                qc.ry(-parameters[index][1], i)

                # measure
                qc.measure(i, i)

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
            hits = self.calculate_qubits_0_hits(histogram)

            # We will assign the data point to the centroid
            # with more 0 hits. E.g. if k = 2 we have hits
            # for the comparison with the first and second
            # centroid. The higher the hits, the closer we
            # are, i.e. the lower the distance. I.e. the
            # hit amount is anti proportional to the distance.
            # Hence, we add a small amount to each hit (to avoid
            # zero division) and take the inverse of it and
            # let the super class calculate the mapping, which
            # will assign to the centroid with minimal distance.
            safe_delta = 50
            for i in range(0, hits.shape[0]):
                distances[index - qubits_for_circuit + i] = 1.0 / (hits[i] + safe_delta)

        return self._calculate_centroid_mapping(data_angles.shape[0], centroid_angles.shape[0], distances)