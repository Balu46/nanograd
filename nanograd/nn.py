import numpy as np
import math
import random
from tensor import Tensor

def Relu(x: Tensor):
    out = Tensor(np.maximum(0, x.data), (x,), 'Relu', label=f'Relu({x.label})')
    return out

class neuron:
    def __init__(self, weights: Tensor, bias: Tensor, activation_function=Relu):
        self.weights = weights
        self.bias = bias
        self.activation = activation_function
        
    
    def __call__(self, x: Tensor):
        # print(type(self.weights))
        # print(type(x))
        return self.activation((self.weights * x).sum() + self.bias)
    
    
    
    
class layer:
    def __init__(self, num_neurons, num_inputs, activation_function=Relu):
        self.neurons = []
        self.num_inputs = num_inputs
        
        for i in range(num_neurons):
            weights = Tensor(np.random.rand(num_inputs), label=f'w{i}')
            bias = Tensor(np.random.rand(1), label=f'b{i}')
            self.neurons.append(neuron(weights, bias, activation_function))
            
    def __call__(self, x: Tensor):
        out = []
        for i in self.neurons:
            out.append(i(x))
        return Tensor(np.array(out), label=f'layer({x.label})')
    
class Mlp:
    def __init__(self, num_layers, activation_function=Relu):
        self.layers = []
        self.activation_function = activation_function
        self.num_layers = num_layers
        
        
        for i in range(len(num_layers) - 1):
            self.layers.append(layer(num_layers[i + 1], num_layers[i], activation_function))
            
    def __call__(self, x: Tensor):
        out = x
        for i in self.layers:
            out = i(out)
        return out
            

if __name__ == "__main__":
    mlp = Mlp([2, 3, 1], Relu)
    x = Tensor(np.array([1, 2]), label='x')
    out = mlp(x)
    
    
    
    
    