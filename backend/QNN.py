import qiskit
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from backend.QNNCircuitGenerator import QNNCircuitGenerator
from backend.QNNcircuitExecutor import CircuitExecutor
import math
import numpy as np

# Adapted from
# Some classes for the hybrid neural network approach

# Based on https://arxiv.org/abs/1912.08278
# and https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
# see also https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb


class NNQuantumCircuit:
    def __init__(self, n_inputs: int, depth: int, backend, shots: int):
        self.circuit_function = QNNCircuitGenerator.genCircuit(n_inputs, depth)
        # self.parameters = self.circuit_function.parameters
        self.backend = backend
        self.shots = shots
        self.n_inputs = n_inputs

    def run(self, inputs: torch.Tensor, weights: torch.Tensor):
        inputs_list = []
        weights_list = []

        for input in inputs:
            inputs_list.append([])
            weights_list.append([])

            for i in range(len(input)):
                inputs_list[-1].append(float(input[i]))
            for i in range(len(weights)):
                weights_list[-1].append(float(weights[i]))

        # run circuit with these parameters
        results, is_statevector = CircuitExecutor.runCircuits(
            self.circuit_function, inputs_list, weights_list, self.n_inputs, self.backend, "", self.shots,
            add_measurements=True)

        # determine counts
        expectations = []
        for i in range(len(inputs)):
            outcome_dict = {}
            if is_statevector:
                temp = results.get_statevector(i)
                outcome_vector = (temp * temp.conj()).real
                # convert outcome_vector to outcome_dict, where key
                # is a basis state and value is the count.
                # Note: the count can be scaled linearly, i.e.,
                # it does not have to be an integer.
                bitstr_size = int(math.log2(len(outcome_vector)))
                for i, _ in enumerate(outcome_vector):
                    bitstr_i = format(i, '0' + str(bitstr_size) + 'b')
                    outcome_dict[bitstr_i] = outcome_vector[i]
            else:
                outcome_dict = results[i]

            # compute counts per qbit
            count_per_bit = {}
            for qbit in range(len(input)):
                count_per_bit[qbit] = [0, 0]
                for key in outcome_dict:
                    count_per_bit[qbit][int(key[qbit])] += outcome_dict[key]

            # determine expectation value per qbit
            # expectation = [(count_per_bit[k][1])/(self.shots*(not is_statevector)+is_statevector)
            expectation = [(count_per_bit[k][1]-count_per_bit[k][0])/(self.shots*(not is_statevector)+is_statevector)
                            for k in count_per_bit]
            expectations.append(expectation)

        return np.array(expectations)


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, weights, quantum_circuit, shift, w2w):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        ctx.w2w = w2w
        expectation_z = quantum_circuit.run(input, weights)
        result = torch.tensor(expectation_z).float()
        ctx.save_for_backward(input, result, weights)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, expectation_z, weights = ctx.saved_tensors

        input_list = np.array(input.tolist())
        n_inputs = input_list.shape[1]
        n_weights = len(weights)

        delta = 2*np.random.default_rng().integers(2, size=n_weights+n_inputs) - 1

        w2w = ctx.w2w if ctx.w2w else n_weights # w2w: weights to wiggle -> number of weights that a gradient is computed for 
        w2w_indices = np.random.choice(range(n_weights), ctx.w2w, replace=False) # randomly select these w2w
        w2w_onehot = [1 for i in range(n_inputs)]+[0 if i not in w2w_indices else 1 for i in range(n_weights)]
        delta *= w2w_onehot

        """ compute input gradients """
        shift_right = input_list + delta[:n_inputs] * ctx.shift
        shift_left = input_list - delta[:n_inputs] * ctx.shift

        shift_right, shift_left = torch.tensor(shift_right), torch.tensor(shift_left)

        expectation_right = ctx.quantum_circuit.run(shift_right, weights)
        expectation_left  = ctx.quantum_circuit.run(shift_left, weights)

        gradient = (torch.tensor(expectation_right) - torch.tensor(expectation_left))*delta[:n_inputs]

        result_gradient = (gradient.float() * grad_output.float())

        """ compute weight gradients """
        weights_right = weights + delta[n_inputs:] * ctx.shift
        weight_left = weights - delta[n_inputs:] * ctx.shift

        weights_expectation_right = ctx.quantum_circuit.run(input_list, weights_right)
        weights_expectation_left  = ctx.quantum_circuit.run(input_list, weight_left)

        gradient = (torch.tensor(weights_expectation_right) - torch.tensor(weights_expectation_left))

        gradient = gradient.float() * grad_output.float()
        weight_gradients = sum([torch.ger(torch.tensor(delta[n_inputs:]).float(), gradient[i]) for i in range(len(gradient))])
        weight_gradients = weight_gradients.T
        return result_gradient, weight_gradients, None, None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, n_inputs, depth, weight_initialization ,backend, shots, shift, w2w):
        super(Hybrid, self).__init__()
        self.quantum_circuit = NNQuantumCircuit(n_inputs, depth, backend, shots)
        weights = self.init_weights(weight_initialization, n_inputs*depth)
        weights = torch.Tensor(weights)
        self.weights = nn.Parameter(weights)
        self.shift = shift
        self.w2w = w2w
        self.epoch = 1

    def update_epoch(self, epoch):
        self.epoch = epoch + 1

    def init_weights(self, weight_init, size):
        if weight_init == "standard_normal":
            return np.random.default_rng().standard_normal(size=size)
        elif weight_init == "uniform":
            return np.random.default_rng().uniform(0, np.pi, size=size)
        elif weight_init == "zero":
            return np.zeros(size)
        raise Exception("Unknown weight initialization technique: {}".format(weight_init))

    def forward(self, input):
        shift = self.shift / (1 + 0.3 * math.log(float(self.epoch), 1.2)) # some decline to the parameter shift
        return HybridFunction.apply(input, self.weights, self.quantum_circuit, shift, self.w2w)

class DressedQNN(nn.Module):
    def __init__(self, dimensions, n_classes, pre_proc_layers, post_proc_layers,
                 quantum_layer_depth, weight_initialization, w2w, shift, backend, shots):
        super(DressedQNN, self).__init__()

        # generate layers of pre processing partial NN
        self.pre_proc_l = []
        if (len(pre_proc_layers)>0):
            self.pre_proc_l.append(nn.Linear(dimensions, pre_proc_layers[0]))
        for i in range(len(pre_proc_layers)-1):
            self.pre_proc_l.append(nn.Linear(pre_proc_layers[i], pre_proc_layers[i+1]))
        self.pre_proc_l = nn.ModuleList(self.pre_proc_l) # properly register the layers

        # generate quantum layer
        self.hybrid_layer = Hybrid(pre_proc_layers[-1], quantum_layer_depth, weight_initialization, backend, shots, shift, w2w)

        # generate layers of post processing partial NN
        self.post_proc_l = []
        for i in range(len(post_proc_layers)-1):
            self.post_proc_l.append(nn.Linear(post_proc_layers[i], post_proc_layers[i+1]))

        # add a final layer to get the required number of outputs for n_classes
        self.post_proc_l.append(nn.Linear(post_proc_layers[-1], n_classes))
        self.post_proc_l = nn.ModuleList(self.post_proc_l) # properly register the layers

    def update_epoch(self, epoch):
        self.hybrid_layer.update_epoch(epoch)

    def forward(self, x):
        # apply classic layers
        if (len(self.pre_proc_l)>0):
            x = F.relu(self.pre_proc_l[0](x))
            # TODO: more activation functions and as parameter to the net
        for i in range(len(self.pre_proc_l)-1):
            x = self.pre_proc_l[i+1](x)

        # apply quantum layer
        x = self.hybrid_layer(x)

        # apply one last classic layer to condense the outcome into one value
        for i in range(len(self.post_proc_l)):
            x = self.post_proc_l[i](x)
        x = F.softmax(x, dim=1)
        return x
