import numpy as np

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, model_params: list, learning_rate: float = 0.01):
        self.model_params = model_params
        self.learning_rate = learning_rate
        
    def step(self):
        for param in self.model_params:
            param.data = param.data - self.learning_rate * param.grad
        
    def zero_grad(self):
        for param in self.model_params:
            param.grad = np.zeros(param.data.shape)
