import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = np.array([[0.01,0.01],
                            [0.01,1],
                            [1,1]])

training_outputs = np.array([[0,0,1]]).T
np.random.seed(1)

weights = np.random.random((2,1))

print('random staring weights')
print(weights)

for iterations in range(60000):

    input_layers = training_inputs
 
    outputs = sigmoid(np.dot(input_layers, weights))
    
    error = training_outputs  - outputs
   
    adjustments = error * sigmoid_derivative(outputs)
   
    weights += np.dot(input_layers.T, adjustments)

print('weights after training:')
print(weights)

print('outputs after training:')
print(outputs)
