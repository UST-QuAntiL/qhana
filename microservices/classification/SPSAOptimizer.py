import numpy as np

class SPSAOptimizer():

    @classmethod
    def initializeOptimization(cls, n_thetas, optimizer_params):
        initial_thetas = np.random.default_rng().standard_normal(n_thetas)

        thetas_plus, thetas_minus, delta = generateDirections(0, initial_thetas, optimizer_params)
        return initial_thetas, thetas_plus, thetas_minus, delta

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