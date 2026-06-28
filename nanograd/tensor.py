import numpy as np
import math
import random

def reduce_grad(grad, target_shape):
    """
    Reduces (sums) the gradient `grad` along dimensions that were broadcasted
    to match the original shape of the parameter `target_shape`.
    """
    if target_shape == ():
        return np.sum(grad)
    
    # If the shapes match exactly, no reduction is needed
    if grad.shape == target_shape:
        return grad
    
    # 1. Sum along excess leading dimensions (if target has fewer dimensions)
    num_extra_dims = grad.ndim - len(target_shape)
    if num_extra_dims > 0:
        grad = np.sum(grad, axis=tuple(range(num_extra_dims)))
        
    # Now grad.ndim is equal to len(target_shape)
    # 2. Sum along dimensions where target_shape has size 1 and grad has size > 1
    axes_to_sum = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_sum.append(i)
            
    if axes_to_sum:
        grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
        
    return grad

class Tensor:
    def __init__(self, data: np.ndarray, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros(self.data.shape)
        self.label = label
        self._backward = lambda: None
        
    def shape(self):
        return self.data.shape
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+', label=f'({self.label}+{other.label})')
        def _backward():
            dself = 1.0 * out.grad
            dother = 1.0 * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*', label=f'({self.label}*{other.label})')
        def _backward():
            dself = other.data * out.grad
            dother = self.data * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward    
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data @ other.data, (self, other), '@', label=f'({self.label}@{other.label})')
        def _backward():
            dself = out.grad @ np.transpose(other.data)
            dother = np.transpose(self.data) @ out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward
        return out
            
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data / other.data, (self, other), '/', label=f'({self.label}/{other.label})')
        def _backward():
            dself = (1.0 / other.data) * out.grad
            dother = (-self.data / (other.data ** 2.0)) * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward    
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other / self
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data ** other.data, (self, other), '**', label=f'({self.label}**{other.label})')
        def _backward():
            dself = other.data * (self.data ** (other.data - 1.0)) * out.grad
            # Compute log safely only where base is strictly positive to prevent RuntimeWarnings and NaNs
            with np.errstate(divide='ignore', invalid='ignore'):
                log_data = np.where(self.data > 0, np.log(np.abs(self.data)), 0.0)
            dother = (self.data ** other.data) * log_data * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other ** self
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp', label=f'exp({self.label})')
        def _backward():
            dself = out.data * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
        out._backward = _backward    
        return out
        
    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum', label=f'sum({self.label})')
        def _backward():
            self.grad = self.grad + out.grad
        out._backward = _backward
        return out
    
    def max(self):
        """
        Computes the maximum value across all elements of the tensor.
        Returns a scalar Tensor.
        """
        # Forward pass: compute the global maximum value
        out = Tensor(self.data.max(), (self,), 'max', label=f'max({self.label})')
        
        def _backward():
            # Backward pass: gradient flows only through the element(s) that achieved the maximum value.
            # Create a boolean mask indicating which elements equal the maximum.
            mask = (self.data == out.data)
            # Propagate the gradient back to those elements.
            self.grad += mask * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-1 * other)
        
    def __rsub__(self, other):
        return other + (-1 * self)
    
    def topo_sort(self):
        visited = set()
        topo = []      
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo
    
    def backward(self):
        if len(self._prev) == 0:
            self.grad = np.ones(self.data.shape) 
            return
        
        topo = self.topo_sort()
        self.grad = np.ones(self.data.shape)
        for i in reversed(topo):
            i._backward()
