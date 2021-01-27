import numpy as np
from qiskit.aqua.algorithms.classifiers.vqc import cost_estimate, return_probabilities
import math

class SPSAOptimizer():
    """
        Optimization (SPSA)
        adapted from qiskit.aqua.components.optimizers.SPSA
        https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.SPSA.html
    """

    @classmethod
    def initializeOptimization(cls, n_thetas, optimizer_params):
        # TODO: Calibrate optimizer parameters

        initial_thetas = np.random.default_rng().standard_normal(n_thetas)

        thetas_plus, thetas_minus, delta = generateDirections(0, initial_thetas, optimizer_params)
        return initial_thetas, thetas_plus, thetas_minus, delta

    @classmethod
    def optimize(cls, results, labels, thetas, delta, iteration, optimizer_params, is_statevector=False):
        results_curr, results_plus, results_minus = np.array_split(results, 3)
        n_data = len(labels)
        n_classes = len(set(labels))

        probs_curr, pred_lbls_curr = computeProbabilities(results_curr, is_statevector, n_data, n_classes)
        probs_plus, pred_lbls_plus = computeProbabilities(results_plus, is_statevector, n_data, n_classes)
        probs_minus, pred_lbls_minus = computeProbabilities(results_minus, is_statevector, n_data, n_classes)
        costs_curr = cost_estimate(probs_curr, labels)
        costs_plus = cost_estimate(probs_plus, labels)
        costs_minus = cost_estimate(probs_minus, labels)

        #print("Current loss: {}, Plus loss: {}, Minus loss: {}".format(costs_curr, costs_plus, costs_minus))

        g_spsa = (costs_plus - costs_minus) * delta / (2.0 * get_c_spsa(iteration, optimizer_params))
        # updated theta
        thetas = thetas - get_a_spsa(iteration, optimizer_params) * g_spsa

        thetas_plus, thetas_minus, delta = generateDirections(iteration, thetas, optimizer_params)

        return thetas, thetas_plus, thetas_minus, delta, costs_curr

def get_a_spsa(iteration, optimizer_params):
    c0, c1, c2, c3, c4 = optimizer_params
    return float(c0) / np.power(iteration + 1 + c4, c2)

def get_c_spsa(iteration, optimizer_params):
    c0, c1, c2, c3, c4 = optimizer_params
    return float(c1) / np.power(iteration + 1, c3)

def gen_delta(size):
    return 2 * np.random.default_rng().integers(2, size=size) - 1

def generateDirections(iteration, thetas, optimizer_params):
    c_spsa = get_c_spsa(iteration, optimizer_params)
    delta = gen_delta(size=thetas.shape[0])

    thetas_plus = thetas + c_spsa * delta
    thetas_minus = thetas - c_spsa * delta
    return thetas_plus, thetas_minus, delta

def computeProbabilities(results, is_statevector, n_data, n_classes):
    """
        compute probabilities from results
    """
    circuit_id = 0
    predicted_probs = []
    predicted_labels = []
    counts = []
    for i in range(n_data):
        if is_statevector:
            temp = results[i].get_statevector()#circuit_id)
            outcome_vector = (temp * temp.conj()).real
            # convert outcome_vector to outcome_dict, where key
            # is a basis state and value is the count.
            # Note: the count can be scaled linearly, i.e.,
            # it does not have to be an integer.
            outcome_dict = {}
            bitstr_size = int(math.log2(len(outcome_vector)))
            for i, _ in enumerate(outcome_vector):
                bitstr_i = format(i, '0' + str(bitstr_size) + 'b')
                outcome_dict[bitstr_i] = outcome_vector[i]
        else:
            outcome_dict = results[i].get_counts()#circuit_id)

        counts.append(outcome_dict)
        circuit_id += 1

    probs = return_probabilities(counts, n_classes)
    predicted_probs.append(probs)
    predicted_labels.append(np.argmax(probs, axis=1))

    if len(predicted_probs) == 1:
        predicted_probs = predicted_probs[0]
    if len(predicted_labels) == 1:
        predicted_labels = predicted_labels[0]

    return predicted_probs, predicted_labels
