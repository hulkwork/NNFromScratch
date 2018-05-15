from data.clusters_data import rand_clusters, rand_linear
import numpy as np
from utils.activation import sigmoid, gaussian, linear
from src.Layer import NeuralNetBackProp

clusters, targets = rand_linear(n_points=300, x_max=1, x_min=-1, y_max=1, y_min=-1)

clusters = np.asarray(clusters)
targets = np.asarray(targets)

transferFunctions = [None, sigmoid, sigmoid, sigmoid]
neural_net_back = NeuralNetBackProp((2, 2, 1, 1), transferFunctions)
neural_net_back.verbose = True
neural_net_back.minError = 1e-4
neural_net_back.verbose_batch = 25000
neural_net_back.maxIterations = neural_net_back.verbose_batch * 10
Error_cust = neural_net_back.trainning(clusters, targets)

clusters, targets = rand_linear(n_points=100, x_max=1, x_min=-1, y_max=1, y_min=-1)
clusters = np.asarray(clusters)
targets = np.asarray(targets)
predict = neural_net_back.feedforwoard(clusters)
error_prediction = np.mean((predict - targets) ** 2)
print("MSE {0:0.10f}".format(error_prediction))
