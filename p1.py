inputs = [1,2,3, 2.5]


weights = [
    [.2, .8, -.5, 1.0],
    [.5, -0.91, .26, -.5],
    [-.26, -.27, .17, .87]
]

biases = [2, 3, .5]

layer_outputs = []  #output of zurrent layer
for neuron_weights, neuron_bias in zip(weights, biases): #zip = combines two lists into list of lists.
    neuron_output = 0   #output of a given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
        
          
print(layer_outputs)