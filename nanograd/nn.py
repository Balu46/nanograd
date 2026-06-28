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
    def model_params(self) -> list:
        return [self.weights, self.bias]
    
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
            params.extend(layer.model_params())
        return params
    
    
def im2col(x: Tensor, kernel_size: tuple, stride: int, padding: int = 0) -> tuple:
    """
    Converts an image input tensor of shape (N, C, H, W) to a column-wise matrix representation.
    This facilitates fast matrix multiplication for convolution operations.
    
    Parameters:
    - x: Input Tensor of shape (N, C, H, W)
    - kernel_size: Tuple containing (kernel_height, kernel_width)
    - stride: Step size for moving the kernel
    - padding: Width of zero-padding added to the input edges
    
    Returns:
    - y: Column-wise reshaped ndarray of shape (N, C * kh * kw, h_out * w_out)
    - out_shape: Tuple with dimensions (N, C, kh, kw, h_out, w_out)
    - origin_shape: Shape of original input (N, C, H, W)
    """
    N, C, _, _ = x.data.shape
    h_in = x.data.shape[2]  
    w_in = x.data.shape[3]
    
    # Calculate output spatial dimensions
    h_out = ((h_in + 2 * padding - kernel_size[0]) // stride) + 1
    w_out = ((w_in + 2 * padding - kernel_size[1]) // stride) + 1
    
    # Pad the input image along spatial dimensions with zeros
    x_pad = np.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    
    # Use NumPy's stride tricks to extract sliding windows without copying memory
    y = np.lib.stride_tricks.as_strided(
        x_pad, 
        (N, C, h_out, w_out, kernel_size[0], kernel_size[1]), 
        (x_pad.strides[0], x_pad.strides[1], x_pad.strides[2] * stride, x_pad.strides[3] * stride, x_pad.strides[2], x_pad.strides[3])
    )

    # Transpose to shape (N, C, kh, kw, h_out, w_out)
    y = y.transpose((0, 1, 4, 5, 2, 3))
    
    # Reshape to (N, C * kh * kw, h_out * w_out)
    y = y.reshape((N, C * kernel_size[0] * kernel_size[1], h_out * w_out))
    
    return y, (N, C, kernel_size[0], kernel_size[1], h_out, w_out), x.data.shape

def col2im(x: np.ndarray, out_shape: tuple, origin_shape: tuple, 
           kernel_size: tuple, stride: int, padding: int = 0) -> np.ndarray:
    """
    Converts a column-wise tensor representation back to an image representation (N, C, H, W).
    This accumulates overlapping gradient contributions (crucial for convolution backward pass).
    
    Parameters:
    - x: Matrix of shape (N, C * kh * kw, h_out * w_out) representing gradients/values
    - out_shape: Strided shape tuple (N, C, kh, kw, h_out, w_out)
    - origin_shape: Shape of original input image (N, C, H, W)
    - kernel_size: Tuple containing (kernel_height, kernel_width)
    - stride: Stride size used in forward pass
    - padding: Padding size used in forward pass
    
    Returns:
    - Image-like ndarray of shape (N, C, H, W)
    """   
    # Reshape to (N, C, kh, kw, h_out, w_out) and transpose back
    x = x.reshape(out_shape)
    x = x.transpose((0, 1, 4, 5, 2, 3))

    # Initialize padded image with zeros
    zero_pad = np.zeros((origin_shape[0], origin_shape[1], origin_shape[2] + 2*padding, origin_shape[3] + 2*padding))
    
    # Accumulate patches into original positions
    for i in range(x.shape[2]):
        for j in range(x.shape[3]):
            h_start = i * stride
            h_end = h_start + kernel_size[0]
            
            w_start = j * stride
            w_end = w_start + kernel_size[1]
            
            # Sum gradients in the overlapping region
            zero_pad[:, :, h_start:h_end, w_start:w_end] += x[:, :, i, j, :, :]
    
    # Remove padding if it was added
    if padding > 0:
        out = zero_pad[:, :, padding:-padding, padding:-padding]
    else:
        out = zero_pad
    
    return out
    
    
class Conv2D:
    """
    Represents a 2D Convolutional neural network layer.
    Uses im2col for forward pass and col2im for backward pass gradient computation.
    """
    def __init__(self, in_channels: int, num_filters: int, kernel_size: tuple, padding: int = 0, stride: int = 1):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.channels = in_channels 
        
        # Initialize weights randomly in [-1, 1], shape: (num_filters, in_channels, kernel_height * kernel_width)
        self.weights = Tensor(np.random.uniform(-1, 1, size=(num_filters, self.channels, np.prod(kernel_size))), label='w')
        # Initialize bias randomly, shape: (1, num_filters)
        self.bias = Tensor(np.random.uniform(-1, 1, size=(1, num_filters)), label='b')
        
    def __call__(self, x: Tensor) -> Tensor:
        # 1. Transform input tensor to column representation
        x_flat, out_shape, origin_shape = im2col(x, self.kernel_size, self.stride, self.padding)
        # Flatten weights to shape: (num_filters, in_channels * kh * kw)
        w_flat = self.weights.data.reshape((self.num_filters, self.channels * np.prod(self.kernel_size)))
        # 2. Perform matrix multiplication: (num_filters, in_channels * kh * kw) @ (N, in_channels * kh * kw, h_out * w_out)
        z_flat = w_flat @ x_flat
        
        N = out_shape[0]
        # Reshape to standard 4D format: (N, num_filters, h_out, w_out)
        z = z_flat.reshape((N, self.num_filters, out_shape[4], out_shape[5]))
        # 3. Add bias: Broadcast bias (1, num_filters, 1, 1) over batch N and spatial dimensions
        z = z + self.bias.data.reshape((1, self.num_filters, 1, 1))
            
        z = Tensor(z, (x, self.weights, self.bias), 'conv2d', label=f'conv2d({x.label})')
        
        def _backward():
            # 1. Gradient with respect to bias
            # Sum z.grad over batch (dim 0) and spatial dimensions (dims 2, 3)
            db = z.grad.sum(axis=(0, 2, 3)) 
            db = db.reshape((1, self.num_filters))
            self.bias.grad += db
            
            # 2. Gradient with respect to weights
            # Reshape upstream gradient to (N, num_filters, h_out * w_out)
            help_z = z.grad.reshape((N, self.num_filters, out_shape[4] * out_shape[5])) 
            # Multiply upstream gradient by transposed x_flat, shape: (num_filters, in_channels * kh * kw)
            dw = help_z @ x_flat.transpose((0, 2, 1))
            # Sum over the batch dimension
            dw = dw.sum(axis=0)
            self.weights.grad += dw.reshape((self.num_filters, self.channels, np.prod(self.kernel_size)))
            
            # 3. Gradient with respect to input x
            # Project gradients back using transposed weights matrix
            dx = w_flat.T @ help_z
            # Transform columns back to image format (handles overlap correctly)
            dx = col2im(dx, out_shape, origin_shape, self.kernel_size, self.stride, self.padding)
            x.grad += dx

        z._backward = _backward
        return z
    def model_params(self) -> list:
        return [self.weights, self.bias]
        
    
class MaxPool2D:
    """
    Represents a 2D Max Pooling layer.
    Reduces spatial dimensions by selecting the maximum value in each window.
    """
    def __init__(self, kernel_size: tuple, stride: int = 2, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        # 1. Reshape image to columns to isolate windows
        x_flat, out_shape, origin_shape = im2col(x, self.kernel_size, self.stride, self.padding)
        C = x.data.shape[1]
        
        # Reshape to separate channels and group elements within each window
        # New shape: (batch_size, C, elements_per_window, num_windows)
        x_flat = x_flat.reshape((x.data.shape[0], C, x_flat.shape[1]//C, x_flat.shape[2]))
        x_flat_shape = x_flat.shape
        
        # 2. Forward pass: Find the maximum value along the window dimension (axis 2)
        x_max = x_flat.max(axis=2)
        # Store index of maximum values for backpropagation
        x_cache = x_flat.argmax(axis=2)
        
        # Reshape to standard 4D layout: (N, C, h_out, w_out)
        y = x_max.reshape((x.data.shape[0], x.data.shape[1], out_shape[4], out_shape[5]))
        y = Tensor(y, (x,), 'max_pool2d', label=f'max_pool2d({x.label})')
        
        def _backward():
            y_grad = y.grad
            # Flatten upstream gradient to (N, C, h_out * w_out)
            y_grad_flaten = y_grad.reshape((y_grad.shape[0], y_grad.shape[1], y_grad.shape[2] * y_grad.shape[3]))
            help_x = np.zeros(x_flat_shape)
            
            # Prepare indices for scatter operation: insert dimensions to align shapes
            x_cache_exp = x_cache[:, :, np.newaxis, :]
            y_grad_exp = y_grad_flaten[:, :, np.newaxis, :]
            
            # Place the gradient elements back to the indices where the maximum values occurred
            np.put_along_axis(help_x, x_cache_exp, y_grad_exp, axis=2)
            
            # Convert strided matrix representation back to image shape (N, C, H, W)
            dx = col2im(help_x, out_shape, origin_shape, self.kernel_size, self.stride, self.padding)
            x.grad += dx
            
        y._backward = _backward
        return y


class Flatten:
    """
    Flattens a multi-dimensional input tensor (typically 4D: batch_size, channels, height, width) 
    to a 2D tensor of shape (batch_size, features), preserving the batch dimension.
    This is commonly used to bridge convolutional/pooling layers and fully-connected layers.
    """
    def __init__(self):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        # Forward pass:
        # Save the original shape for the backward pass reconstruction.
        original_shape = x.data.shape
        # Flatten all dimensions except the batch dimension (axis 0).
        x_flat_data = x.data.reshape((original_shape[0], original_shape[1] * original_shape[2] * original_shape[3]))
        
        # Package the result in a Tensor tracking the input 'x' as a dependency.
        x_flat = Tensor(x_flat_data, (x,), 'flatten', label=f'flatten({x.label})')
        
        def _backward():
            # Backward pass:
            # Reshape the upstream gradient back to the original shape of the input tensor.
            x_flat_grad = x_flat.grad.reshape(original_shape)
            # Accumulate the gradients back to the input tensor.
            x.grad += x_flat_grad
            
        x_flat._backward = _backward
        return x_flat




