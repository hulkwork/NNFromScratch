from data import minst
from src.Layer import NeuralNetBackProp
from utils.activation import sigmoid,gaussian, linear
import numpy as np
from collections import Counter

minst.get_mminst()

(train,target_train, test, target_test, header) = minst.get_data(train_size=500,test_size=10)
uniq = []
for item in target_train:
    uniq.append(np.argmax(item) )
print(Counter(uniq))
#raw_input()
n_input = len(train[0])
print("Input size %d" % n_input)
transferFunctions = [None, linear,sigmoid]
NN = NeuralNetBackProp((n_input, 300 , len(target_train[0])), transferFunctions)
maxIterations = 2500 * 3
minError = 1e-5
batch = 1
NN.verbose = True
NN.minError = 1e-4
NN.verbose_batch = 25
NN.maxIterations = NN.verbose_batch * 10
Error = NN.trainning(train, target_train)

Output = NN.feedforwoard(test)
print 'Input \tOutput \t\tTarget'
for i in range(test.shape[0]):
    print '{0}\t {1} \t{2}'.format(i, np.argmax(Output[i]), np.argmax(target_test[i]) )

