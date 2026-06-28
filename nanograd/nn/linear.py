from nanograd.core.backend import get_xp
from nanograd.core.tensor import Tensor
from nanograd.nn.module import Module, Parameter
from nanograd.nn.activations import relu

class Layer(Module):
    """
    Represents a single fully-connected neural network layer.
    """
    def __init__(self, num_neurons: int, num_inputs: int, activation_function=relu):
        import numpy as np
        super().__init__()
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = Parameter(np.random.uniform(-1, 1, size=(num_inputs, num_neurons)), label='w')
        self.bias = Parameter(np.random.uniform(-1, 1, size=(1, num_neurons)), label='b')
        
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        out = (x @ self.weights) + self.bias
        if self.activation_function is not None:
            out = self.activation_function(out)
        return out


# Alias Linear to Layer for PyTorch compatibility
Linear = Layer


class MLP(Module):
    """
    Represents a Multi-Layer Perceptron (MLP) neural network.
    """
    def __init__(self, num_layers: list, activation_function=relu):
        super().__init__()
        self.activation_function = activation_function
        self.num_layers = num_layers
        self.layers = []
        
        for i in range(len(num_layers) - 1):
            self.layers.append(Layer(num_layers[i + 1], num_layers[i], activation_function))
            
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        for layer in self.layers:
            x = layer(x)
        return x


class Flatten(Module):
    """
    Flattens a multi-dimensional input tensor to 2D (batch_size, features).
    """
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        original_shape = x.data.shape
        x_flat_data = x.data.reshape((original_shape[0], int(xp.prod(xp.array(original_shape[1:])))))
        x_flat = Tensor(x_flat_data, (x,), 'flatten', label=f'flatten({x.label})')
        
        def _backward():
            xp = get_xp(x.data)
            x.grad += x_flat.grad.reshape(original_shape)
            
        x_flat._backward = _backward
        return x_flat
