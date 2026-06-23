import numpy as np
import math
import random
from .tensor import Tensor

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, model_params: list, learning_rate: float = 0.0):
        self.model_params = model_params
        self.learning_rate = learning_rate
        
    def step(self):
        """
        Updates the values of parameters using gradient descent step.
        """
        for param in self.model_params:
            param.data = param.data - self.learning_rate * param.grad
        
    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero.
        """
        for param in self.model_params:
            param.grad = np.zeros(param.data.shape)
