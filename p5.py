import math
import numpy as np
#Softmax Activation
#input -> Exponentiate -> Normalize -> Output
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, .2],
                 [1.41, 1.051, .026]
                ]
#Exponentiation
#Work on indiviaul value level
exp_values = np.exp(layer_outputs)

#Normalization
#axis = 0 -> sum of cols, axis = 1 -> sum of rows
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims = True)
print(norm_values)

