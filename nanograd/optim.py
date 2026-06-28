import numpy as np
import math
import random
from .tensor import Tensor

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, model_params: list, learning_rate: float = 0.01):
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


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.
    Combines ideas from RMSProp and Momentum to adapt learning rates for each parameter.
    """
    def __init__(self, model_params: list, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.t = 0
        self.model_params = model_params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize 1st moment vector (m) and 2nd moment vector (v) to zeros
        self.m = []
        self.v = []
        for param in self.model_params:
            self.m.append(np.zeros(param.data.shape))
            self.v.append(np.zeros(param.data.shape))
        
    def step(self):
        """
        Updates the values of parameters using Adam's optimization step.
        """
        self.t += 1
        
        for i in range(len(self.model_params)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.model_params[i].grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (self.model_params[i].grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter values
            self.model_params[i].data = self.model_params[i].data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
    def zero_grad(self):
        """
        Resets the gradients of all parameters to zero.
        """
        for param in self.model_params:
            param.grad = np.zeros(param.data.shape)