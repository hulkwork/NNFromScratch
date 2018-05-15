import numpy as np
from utils.activation import linear, sigmoid
import pickle
np.random.seed(2016)


class NeuralNetBackProp:
    number_of_layers = 0
    shape = None
    weight = []
    biais = []
    maxIterations = 2500 * 3
    minError = 1e-5
    learningRate = 0.2
    momentum = 0.5
    verbose = True
    verbose_batch = 2500

    def __init__(self, number_of_nodes_layer, activation_functions=None):

        self.number_of_layers = len(number_of_nodes_layer) - 1
        self.shape = number_of_nodes_layer
        if activation_functions is None:
            layer_activation_function = []
            for i in range(self.number_of_layers):
                if i == self.number_of_layers - 1:
                    layer_activation_function.append(linear)
                else:
                    layer_activation_function.append(sigmoid)
        else:
            if len(number_of_nodes_layer) != len(activation_functions):
                raise ValueError("Number of transfer functions must match the number of layers: minus input layer")
            elif activation_functions[0] is not None:
                raise ValueError("The Input layer doesn't need a a transfer function: give it [None,...]")
            else:
                layer_activation_function = activation_functions[1:]
        self.activation_functions = layer_activation_function
        self.previous_weight_delta = []
        self.previous_biais_delta = []
        for (l1, l2) in zip(number_of_nodes_layer[:-1], number_of_nodes_layer[1:]):
            self.weight.append(np.random.normal(scale=0.1, size=(l2, l1)))
            self.previous_weight_delta.append(np.zeros((l2, l1)))
            self.biais.append(np.ones([l2, 1]))
            self.previous_biais_delta.append(np.ones([l2, 1]))

    def feedforwoard(self, X):
        self.layer_input = []
        self.layer_output = []
        for index in range(self.number_of_layers):
            if index == 0:
                layer_input = self.weight[0].dot(X.T) + self.biais[0]
            else:
                layer_input = self.weight[index].dot(self.layer_output[-1]) + self.biais[index]
            self.layer_input.append(layer_input)
            self.layer_output.append(self.activation_functions[index](layer_input))
        return self.layer_output[-1].T

    def back_propagation(self, X, target, learningRate=0.2, momentum=0.5):
        delta = []
        self.feedforwoard(X)
        for index in reversed(range(self.number_of_layers)):
            if index == self.number_of_layers - 1:
                output_delta = self.layer_output[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.activation_functions[index](self.layer_input[index], True))
            else:
                delta_pullback = self.weight[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback * self.activation_functions[index](self.layer_input[index], True))
        for index in range(self.number_of_layers):
            delta_index = self.number_of_layers - 1 - index
            if index == 0:
                layer_output = X.T
            else:
                layer_output = self.layer_input[index - 1]
            this_weight_delta = layer_output.dot(delta[delta_index].T)
            this_biais_delta = np.sum(delta[delta_index].T, axis=0)
            weight_delta = learningRate * this_weight_delta.T + momentum * self.previous_weight_delta[index]
            biais_delta = learningRate * this_biais_delta.T + momentum * self.previous_biais_delta[index]
            self.weight[index] -= weight_delta
            biais_delta = np.sum(biais_delta, axis=0, keepdims=True)
            self.biais[index] -= biais_delta.T
            self.previous_weight_delta[index] = weight_delta
            self.previous_biais_delta[index] = biais_delta
        return error

    def trainning(self, X, y):
        Error = -1.0
        for i in range(self.maxIterations + 1):
            Error = self.back_propagation(X, y, learningRate=self.learningRate, momentum=self.momentum)
            if self.verbose:
                if i % self.verbose_batch == 0:
                    print("Iteration {0}\tError: {1:0.6f}".format(i, Error))

            if Error <= self.minError:
                print("Minimum error reached at iteration {0}".format(i))
                break

        print("Iteration {0}\tError: {1:0.6f}".format(i, Error))
        return Error

    def save(self, pathfile):
        # TODO: save all params into pickle files
        pass

    def load(self, pathfile):
        # TODO: load all params from pickle files
        pass




