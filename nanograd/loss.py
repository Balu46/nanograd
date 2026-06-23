import numpy as np
import math
import random
from .tensor import Tensor

class MSE:
    """
    Mean Squared Error (MSE) loss function.
    Calculates the mean squared difference between predictions and targets.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        n = y_pred.data.shape[0]
        loss = ((y_pred - y_true) ** 2).sum() / n
        return loss
    
    

    
    
class SoftmaxCrossEntropy:
    """
    Softmax Cross Entropy loss function.
    Calculates the cross entropy loss between logits (y_pred) and target probability distributions (y_true)
    using an integrated softmax function for numerical stability.
    """
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Forward pass:
        # 1. Softmax activation with numerical stability trick
        n = y_pred.data.shape[0]
        y_max = y_pred.data.max(axis=1, keepdims=True)
        y_prim = y_pred.data - y_max
        exp_y = np.exp(y_prim)
        sum_exp_y = exp_y.sum(axis=1, keepdims=True)
        probs = exp_y / sum_exp_y
        
        # 2. Cross Entropy Loss calculation
        epsilon = 1e-10  # small value to avoid log(0)
        loss = -(y_true.data * np.log(probs + epsilon)).sum()
        loss_total = loss / n
        
        # 3. Wrap the result in a Tensor tracking y_pred and y_true as children
        loss_total = Tensor(loss_total, (y_pred, y_true), 'softmax_cross_entropy', label=f'softmax_cross_entropy({y_pred.label}, {y_true.label})')
        
        def _backward():
            # Backward pass:
            # The derivative of the batch-averaged cross-entropy loss with respect to logits (y_pred) is:
            # d(loss_total)/d(y_pred) = 1/N * (probs - y_true) * d(loss_total)/d(loss_total)
            # where N is the batch size and probs is the computed softmax probability.
            n = y_pred.data.shape[0]
            y_pred.grad += (1/n) * (probs - y_true.data) * loss_total.grad
            
        loss_total._backward = _backward
    
        return loss_total
    

