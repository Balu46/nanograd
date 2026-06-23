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
        x.grad += reduce_grad(dx, x.data.shape)
        
    out._backward = _backward
    return out

def softmax(x: Tensor) -> Tensor:
    """
    Applies the softmax activation function along the class axis (axis=1) 
    for a 2D input tensor of shape (batch_size, num_classes).
    """
    # Forward pass:
    # 1. Shift inputs for numerical stability to prevent potential overflow/underflow.
    x_max = x.data.max(axis=1, keepdims=True)
    x_prim = x.data - x_max
    # 2. Compute exponentials and sum them along the class dimension.
    exp_x = np.exp(x_prim)
    sum_exp_x = exp_x.sum(axis=1, keepdims=True)
    # 3. Calculate probability distributions.
    probs = Tensor(exp_x / sum_exp_x, (x,), 'softmax', label=f'softmax({x.label})')
    
    def _backward():
        # Backward pass:
        # For an output vector p = softmax(x), the gradient of a loss L with respect to x is:
        # dx_j = p_j * (g_j - sum_k (p_k * g_k))
        # where g = probs.grad is the upstream gradient.
        #
        # Calculate the inner sum: dp_b = sum_k (p_{b,k} * g_{b,k}) for each sample in the batch.
        dp = (probs.data * probs.grad).sum(axis=1, keepdims=True)
        # Calculate (g_j - dp_b)
        help = probs.grad - dp
        # Multiply by p_j and add to the input gradient (accumulating gradients).
        x.grad += probs.data * help
        
    probs._backward = _backward
    
    return probs
    

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