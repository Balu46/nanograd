import numpy as np
import math
import random
from .tensor import Tensor, reduce_grad

def relu(x: Tensor) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) activation function element-wise.
    """
    out = Tensor(np.maximum(0, x.data), (x,), 'relu', label=f'relu({x.label})')
    
    def _backward():
        dx = (x.data > 0) * out.grad
        x.grad = x.grad + reduce_grad(dx, x.data.shape)
        
    out._backward = _backward
    return out

class Layer:
    """
    Represents a single fully-connected neural network layer.
    """
    def __init__(self, num_neurons: int, num_inputs: int, activation_function=relu):
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        # Randomly initialize weights and biases in the range [-1, 1]
        self.weights = Tensor(np.random.uniform(-1, 1, size=(num_inputs, num_neurons)), label='w')
        self.bias = Tensor(np.random.uniform(-1, 1, size=(1, num_neurons)), label='b')
        
    def __call__(self, x: Tensor) -> Tensor:
        # Linear transform followed by activation function
        x = (x @ self.weights) + self.bias
        x = self.activation_function(x)
        return x
    
class MLP:
    """
    Represents a Multi-Layer Perceptron (MLP) neural network.
    """
    def __init__(self, num_layers: list, activation_function=relu):
        self.activation_function = activation_function
        self.num_layers = num_layers
        self.layers = []
        
        # Instantiate layers matching the size configurations
        for i in range(len(num_layers) - 1):
            self.layers.append(Layer(num_layers[i + 1], num_layers[i], activation_function))
            
    def __call__(self, x: Tensor) -> Tensor:
        # Feed-forward through all layers
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self) -> list:
        # Collect and return weights and biases from all layers
        params = []
        for layer in self.layers:
            params.append(layer.weights)
            params.append(layer.bias)
        return params