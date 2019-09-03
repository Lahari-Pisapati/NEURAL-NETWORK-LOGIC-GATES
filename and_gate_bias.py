import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

training_outputs = np.array([[0,0,0,1]]).T
np.random.seed(1)

weights = np.random.random((2,1))
bias = 0
print('random staring weights')
print(weights)

for iterations in range(60000):

    input_layers = training_inputs
 
    outputs = sigmoid(np.dot(input_layers, weights)+ bias)
    
    error = training_outputs  - outputs
   
    adjustments = error * sigmoid_derivative(outputs)
   
    weights += np.dot(input_layers.T, adjustments)
    bias += np.sum(adjustments)
print('weights after training:')
print(weights)

print('outputs after training:')
print(outputs)

x1 = float(input('enter the value x1: '))
x2 = float(input('enter the value x2: '))

test_inputs = np.array([[x1,x2]])
print(test_inputs)
test_output = sigmoid(np.dot(test_inputs, weights)+ bias)
print(test_output)
