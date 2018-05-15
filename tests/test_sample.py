import numpy as np
from utils.activation import sigmoid, linear
from src.Layer import NeuralNetBackProp

Input = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
Target = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
transferFunctions = [None, sigmoid, linear, sigmoid]
neural_net_back = NeuralNetBackProp((2, 2, 3, 2), transferFunctions)
neural_net_back.verbose = False
neural_net_back.minError = 1e-4
neural_net_back.verbose_batch = 2500
neural_net_back.maxIterations = neural_net_back.verbose_batch * 10
Error_cust = neural_net_back.trainning(Input, Target)

Output = neural_net_back.feedforwoard(Input)
print 'Input \tOutput \t\tTarget'
for i in range(Input.shape[0]):
    print '{0}\t {1} \t{2}'.format(Input[i], Output[i], Target[i])
