import numpy as np
import math
import random

from .backend import get_xp, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def reduce_grad(grad, target_shape):
    """
    Reduces (sums) the gradient `grad` along dimensions that were broadcasted
    to match the original shape of the parameter `target_shape`.
    """
    xp = get_xp(grad)
    if target_shape == ():
        return xp.sum(grad)
    
    # If the shapes match exactly, no reduction is needed
    if grad.shape == target_shape:
        return grad
    
    # 1. Sum along excess leading dimensions (if target has fewer dimensions)
    num_extra_dims = grad.ndim - len(target_shape)
    if num_extra_dims > 0:
        grad = xp.sum(grad, axis=tuple(range(num_extra_dims)))
        
    # Now grad.ndim is equal to len(target_shape)
    # 2. Sum along dimensions where target_shape has size 1 and grad has size > 1
    axes_to_sum = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_sum.append(i)
            
    if axes_to_sum:
        grad = xp.sum(grad, axis=tuple(axes_to_sum), keepdims=True)
        
    return grad

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', device='cpu'):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        xp = get_xp(self.data)
        self.grad = xp.zeros(self.data.shape)
        self.label = label
        self._backward = lambda: None
        self.device = device
    
    def to(self, device):
        """
        Moves the Tensor to the specified device ('cpu' or 'cuda').
        Returns a new Tensor with the data transferred to the target device.
        """
        # If the tensor is already on the requested device, do nothing
        if device == self.device:
            return self

        if device == 'cuda':
            if not CUPY_AVAILABLE:
                raise RuntimeError("CuPy is not installed or no GPU was detected.")
            # cp.asarray physically copies data from CPU RAM to GPU VRAM
            new_data = cp.asarray(self.data)
            
        elif device == 'cpu':
            # cp.asnumpy brings data back from the GPU to the CPU
            if self.device == 'cuda':
                new_data = cp.asnumpy(self.data)
            else:
                new_data = np.asarray(self.data)
        else:
            raise ValueError(f"Unknown device specified: {device}. Use 'cpu' or 'cuda'.")

        # Create a new Tensor object with the transferred data.
        # We pass the same graph connections (_prev, _op) to keep backprop working.
        new_tensor = Tensor(
            data=new_data, 
            _children=self._prev, 
            _op=self._op, 
            label=self.label, 
            device=device
        )
        
        # If gradients have already been accumulated, they must be moved too
        if self.grad is not None:
            if device == 'cuda':
                new_tensor.grad = cp.asarray(self.grad)
            elif device == 'cpu':
                if self.device == 'cuda':
                    new_tensor.grad = cp.asnumpy(self.grad)
                else:
                    new_tensor.grad = np.asarray(self.grad)

        return new_tensor

    # --- Optional but highly recommended shortcuts ---
    
    def cuda(self):
        """Shortcut to move the tensor to GPU."""
        return self.to('cuda')

    def cpu(self):
        """Shortcut to move the tensor to CPU."""
        return self.to('cpu')
    
    def shape(self):
        return self.data.shape
        
    def __add__(self, other):
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
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
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
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
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        out = Tensor(self.data @ other.data, (self, other), '@', label=f'({self.label}@{other.label})')
        def _backward():
            dself = out.grad @ other.data.T
            dother = self.data.T @ out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward
        return out
            
    def __truediv__(self, other):
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        out = Tensor(self.data / other.data, (self, other), '/', label=f'({self.label}/{other.label})')
        def _backward():
            dself = (1.0 / other.data) * out.grad
            dother = (-self.data / (other.data ** 2.0)) * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward    
        return out
    
    def __rtruediv__(self, other):
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        return other / self
    
    def __pow__(self, other):
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        out = Tensor(self.data ** other.data, (self, other), '**', label=f'({self.label}**{other.label})')
        def _backward():
            xp = get_xp(self.data)
            dself = other.data * (self.data ** (other.data - 1.0)) * out.grad
            # Compute log safely only where base is strictly positive to prevent RuntimeWarnings and NaNs
            if xp is np:
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_data = xp.where(self.data > 0, xp.log(xp.abs(self.data)), 0.0)
            else:
                log_data = xp.where(self.data > 0, xp.log(xp.abs(self.data)), 0.0)
            dother = (self.data ** other.data) * log_data * out.grad
            self.grad = self.grad + reduce_grad(dself, self.data.shape)
            other.grad = other.grad + reduce_grad(dother, other.data.shape)
        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        xp = get_xp(self.data)
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        return other ** self
    
    def exp(self):
        xp = get_xp(self.data)
        out = Tensor(xp.exp(self.data), (self,), 'exp', label=f'exp({self.label})')
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
        xp = get_xp(self.data)
        if len(self._prev) == 0:
            self.grad = xp.ones(self.data.shape) 
            return
        
        topo = self.topo_sort()
        self.grad = xp.ones(self.data.shape)
        for i in reversed(topo):
            i._backward()
