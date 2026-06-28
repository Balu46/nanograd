import numpy as np
from nanograd.core.tensor import Tensor
from nanograd.nn.module import Module

class MSE(Module):
    """
    Mean Squared Error (MSE) loss function.
    """
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        n = y_pred.data.shape[0]
        loss = ((y_pred - y_true) ** 2).sum() / n
        return loss


class SoftmaxCrossEntropy(Module):
    """
    Softmax Cross Entropy loss function.
    """
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        n = y_pred.data.shape[0]
        y_max = y_pred.data.max(axis=1, keepdims=True)
        y_prim = y_pred.data - y_max
        exp_y = np.exp(y_prim)
        sum_exp_y = exp_y.sum(axis=1, keepdims=True)
        probs = exp_y / sum_exp_y
        
        epsilon = 1e-10
        loss = -(y_true.data * np.log(probs + epsilon)).sum()
        loss_total = loss / n
        
        loss_total = Tensor(loss_total, (y_pred, y_true), 'softmax_cross_entropy', label=f'softmax_cross_entropy({y_pred.label}, {y_true.label})')
        
        def _backward():
            y_pred.grad += (1/n) * (probs - y_true.data) * loss_total.grad
            
        loss_total._backward = _backward
        return loss_total


# Alias CrossEntropy to SoftmaxCrossEntropy
CrossEntropy = SoftmaxCrossEntropy
