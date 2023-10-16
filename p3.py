import numpy as np
np.random.seed(0)
#Objects


X = [[1, 2, 3, 2.5],
          [2.0, 5.0,-1.0, 2.0],
          [-1.5, 2.7, 3.3, -.8]
]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) #np.zeros = Returns a new array of given shape and type, filled with zeros
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2) #First input must be last input of previous (matrix multiplication)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

