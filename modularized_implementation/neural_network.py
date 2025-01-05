import random

from modularized_implementation.micro_grad import Value
from typing import List, Union

class Perceptron:
    def __init__(self, num_inputs: int):
        self.weights = [Value(data=random.uniform(a=0, b=1), label=f"weight") for i in range(num_inputs)]
        self.bias = Value(data=0.0, label="bias")

    def __call__(self, inputs: List[Union[Value, float, int]]) -> Value:
        weighted_sum = self.bias
        for input_value, weight in zip(inputs, self.weights):
            weighted_sum += input_value * weight
        output = weighted_sum.tanh()
        return output

    def get_parameters(self) -> List[Value]:
        return self.weights + [self.bias]


class Layer:
    def __init__(self, num_inputs: int, num_neurons: int):
        self.neurons = [Perceptron(num_inputs=num_inputs) for _ in range(num_neurons)]
    
    def __call__(self, inputs: List[Union[Value, float, int]]) -> List[Value]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(inputs))
        return outputs       

    def get_parameters(self) -> List[Value]:
        layer_parameters = []
        for neuron in self.neurons:
            layer_parameters.extend(neuron.get_parameters())
        return layer_parameters


class MultiLayerPerceptron:
    def __init__(self, num_inputs: int, num_neurons_per_layer: List[int]):
        self.layers = [Layer(num_inputs=num_inputs, num_neurons=num_neurons_per_layer[0])]
        for i in range(1, len(num_neurons_per_layer)):
            self.layers.append(Layer(num_inputs=num_neurons_per_layer[i-1], num_neurons=num_neurons_per_layer[i]))
        
    def __call__(self, inputs: List[Union[Value, float, int]]) -> List[Value]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def get_parameters(self) -> List[Value]:
        model_parameters = []
        for layer in self.layers:
            model_parameters.extend(layer.get_parameters())
        return model_parameters