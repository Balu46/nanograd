from nanograd.core.backend import get_xp
from nanograd.core.tensor import Tensor
from nanograd.nn.module import Module, Parameter

def im2col(x: Tensor, kernel_size: tuple, stride: int, padding: int = 0) -> tuple:
    """
    Converts an image input tensor of shape (N, C, H, W) to a column-wise matrix representation.
    """
    xp = get_xp(x.data)
    N, C, _, _ = x.data.shape
    h_in = x.data.shape[2]  
    w_in = x.data.shape[3]
    
    # Calculate output spatial dimensions
    h_out = ((h_in + 2 * padding - kernel_size[0]) // stride) + 1
    w_out = ((w_in + 2 * padding - kernel_size[1]) // stride) + 1
    
    # Pad the input image along spatial dimensions with zeros
    x_pad = xp.pad(x.data, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    
    # Use NumPy's stride tricks to extract sliding windows without copying memory
    y = xp.lib.stride_tricks.as_strided(
        x_pad, 
        (N, C, h_out, w_out, kernel_size[0], kernel_size[1]), 
        (x_pad.strides[0], x_pad.strides[1], x_pad.strides[2] * stride, x_pad.strides[3] * stride, x_pad.strides[2], x_pad.strides[3])
    )

    # Transpose to shape (N, C, kh, kw, h_out, w_out)
    y = y.transpose((0, 1, 4, 5, 2, 3))
    
    # Reshape to (N, C * kh * kw, h_out * w_out)
    y = y.reshape((N, C * kernel_size[0] * kernel_size[1], h_out * w_out))
    
    return y, (N, C, kernel_size[0], kernel_size[1], h_out, w_out), x.data.shape


def col2im(x, out_shape: tuple, origin_shape: tuple, 
           kernel_size: tuple, stride: int, padding: int = 0):
    """
    Converts a column-wise tensor representation back to an image representation (N, C, H, W).
    """   
    xp = get_xp(x)
    # Reshape to (N, C, kh, kw, h_out, w_out) and transpose back
    x = x.reshape(out_shape)
    x = x.transpose((0, 1, 4, 5, 2, 3))

    # Initialize padded image with zeros
    zero_pad = xp.zeros((origin_shape[0], origin_shape[1], origin_shape[2] + 2*padding, origin_shape[3] + 2*padding))
    
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


class Conv2D(Module):
    """
    Represents a 2D Convolutional neural network layer.
    """
    def __init__(self, in_channels: int, num_filters: int, kernel_size: tuple, padding: int = 0, stride: int = 1):
        import numpy as np
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.channels = in_channels 
        
        self.weights = Parameter(np.random.uniform(-1, 1, size=(num_filters, in_channels, np.prod(kernel_size))), label='w')
        self.bias = Parameter(np.random.uniform(-1, 1, size=(1, num_filters)), label='b')
        
    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        x_flat, out_shape, origin_shape = im2col(x, self.kernel_size, self.stride, self.padding)
        w_flat = self.weights.data.reshape((self.num_filters, self.channels * int(xp.prod(xp.array(self.kernel_size)))))
        z_flat = w_flat @ x_flat
        
        N = out_shape[0]
        z = z_flat.reshape((N, self.num_filters, out_shape[4], out_shape[5]))
        z = z + self.bias.data.reshape((1, self.num_filters, 1, 1))
            
        z = Tensor(z, (x, self.weights, self.bias), 'conv2d', label=f'conv2d({x.label})')
        
        def _backward():
            xp = get_xp(x.data)
            db = z.grad.sum(axis=(0, 2, 3)) 
            db = db.reshape((1, self.num_filters))
            self.bias.grad += db
            
            help_z = z.grad.reshape((N, self.num_filters, out_shape[4] * out_shape[5])) 
            dw = help_z @ x_flat.transpose((0, 2, 1))
            dw = dw.sum(axis=0)
            self.weights.grad += dw.reshape((self.num_filters, self.channels, int(xp.prod(xp.array(self.kernel_size)))))
            
            dx = w_flat.T @ help_z
            dx = col2im(dx, out_shape, origin_shape, self.kernel_size, self.stride, self.padding)
            x.grad += dx

        z._backward = _backward
        return z


class MaxPool2D(Module):
    """
    Represents a 2D Max Pooling layer.
    """
    def __init__(self, kernel_size: tuple, stride: int = 2, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        xp = get_xp(x.data)
        x_flat, out_shape, origin_shape = im2col(x, self.kernel_size, self.stride, self.padding)
        C = x.data.shape[1]
        
        x_flat = x_flat.reshape((x.data.shape[0], C, x_flat.shape[1]//C, x_flat.shape[2]))
        x_flat_shape = x_flat.shape
        
        x_max = x_flat.max(axis=2)
        x_cache = x_flat.argmax(axis=2)
        
        y = x_max.reshape((x.data.shape[0], x.data.shape[1], out_shape[4], out_shape[5]))
        y = Tensor(y, (x,), 'max_pool2d', label=f'max_pool2d({x.label})')
        
        def _backward():
            xp = get_xp(x.data)
            y_grad = y.grad
            y_grad_flaten = y_grad.reshape((y_grad.shape[0], y_grad.shape[1], y_grad.shape[2] * y_grad.shape[3]))
            help_x = xp.zeros(x_flat_shape)
            
            x_cache_exp = x_cache[:, :, xp.newaxis, :]
            y_grad_exp = y_grad_flaten[:, :, xp.newaxis, :]
            
            xp.put_along_axis(help_x, x_cache_exp, y_grad_exp, axis=2)
            dx = col2im(help_x, out_shape, origin_shape, self.kernel_size, self.stride, self.padding)
            x.grad += dx
            
        y._backward = _backward
        return y
