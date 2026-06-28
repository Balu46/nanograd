from nanograd.core.backend import get_xp
from nanograd.core.tensor import Tensor, reduce_grad
from nanograd.nn.module import Module

def relu(x: Tensor) -> Tensor:
    """
    Applies the rectified linear unit (ReLU) activation function element-wise.
    """
    xp = get_xp(x.data)
    out = Tensor(xp.maximum(0, x.data), (x,), 'relu', label=f'relu({x.label})')
    
    def _backward():
        xp = get_xp(x.data)
        dx = (x.data > 0) * out.grad
        x.grad += reduce_grad(dx, x.data.shape)
        
    out._backward = _backward
    return out


def softmax(x: Tensor) -> Tensor:
    """
    Applies the softmax activation function along the class axis (axis=1) 
    for a 2D input tensor of shape (batch_size, num_classes).
    """
    xp = get_xp(x.data)
    x_max = x.data.max(axis=1, keepdims=True)
    x_prim = x.data - x_max
    exp_x = xp.exp(x_prim)
    sum_exp_x = exp_x.sum(axis=1, keepdims=True)
    probs = Tensor(exp_x / sum_exp_x, (x,), 'softmax', label=f'softmax({x.label})')
    
    def _backward():
        xp = get_xp(x.data)
        dp = (probs.data * probs.grad).sum(axis=1, keepdims=True)
        help_val = probs.grad - dp
        x.grad += probs.data * help_val
        
    probs._backward = _backward
    return probs


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        return relu(x)


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        return softmax(x)
