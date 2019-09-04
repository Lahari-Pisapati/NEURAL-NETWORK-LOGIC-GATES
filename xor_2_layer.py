import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = np.array([[0.01,0.01],
                            [0.01,1],
                            [1,0.01],
                            [1,1]])

training_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)

weights_1 = np.random.random((2,2))
weights_2 = np.random.random((2,1))

bias_1 = 0
bias_2 = 0

for iterations in range(60000):

    input_layer = training_inputs

    layer_1 = sigmoid(np.dot(input_layer, weights_1)+ bias_1)
    
    layer_2 = sigmoid(np.dot(layer_1, weights_2)+ bias_2)
    error =(training_outputs - layer_2)
    
    adjust_2 = 1/4 * error * sigmoid_derivative(layer_2)   
    weights_2 += np.dot(layer_1.T,adjust_2)
    bias_2 += np.sum(adjust_2)
    adjust_1 = 1/4 * error * sigmoid_derivative(layer_2) * sigmoid_derivative(layer_1) * weights_2.T
    weights_1 += np.dot(input_layer.T, adjust_1)
    bias_1 += np.sum(adjust_1)
    
print('weights:')
print(weights_1)
print(weights_2)
print('bias:')
print(bias_1)
print(bias_2)
print('output:')
print(layer_2)
