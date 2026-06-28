import pytest
import numpy as np
import torch
from nanograd.tensor import Tensor
from nanograd.nn import relu, softmax, Layer, MLP, Conv2D, MaxPool2D, Flatten
from nanograd.loss import MSE, SoftmaxCrossEntropy
from nanograd.optim import SGD, Adam
import torch.nn.functional as F

def test_relu():
    """Test relu activation function forward and backward passes."""
    x_data = np.array([[-1.0, 0.0, 2.0], [3.0, -0.5, 1.0]])
    x_nano = Tensor(x_data)
    out_nano = relu(x_nano)
    out_nano.backward()

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    out_pt = torch.relu(x_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy())
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy())

def test_layer():
    """Test individual Layer forward and backward passes."""
    np.random.seed(42)
    l = Layer(num_neurons=3, num_inputs=2)
    
    # Set deterministic weights and bias for consistency
    w_data = np.array([[0.5, -0.1, 0.8], [0.2, 0.4, -0.3]])
    b_data = np.array([[0.1, 0.2, -0.1]])
    l.weights = Tensor(w_data)
    l.bias = Tensor(b_data)

    x_data = np.array([[1.0, -1.0]])
    x_nano = Tensor(x_data)
    out_nano = l(x_nano)
    out_nano.backward()

    # PyTorch counterpart
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    w_pt = torch.tensor(w_data, requires_grad=True, dtype=torch.double)
    b_pt = torch.tensor(b_data, requires_grad=True, dtype=torch.double)
    
    out_pt = torch.relu(x_pt @ w_pt + b_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(l.weights.grad, w_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(l.bias.grad, b_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5)

def test_mlp():
    """Test multi-layer perceptron (MLP) forward and backward passes."""
    np.random.seed(42)
    mlp = MLP([2, 3, 1])
    
    # Set deterministic weights for the MLP
    w1 = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
    b1 = np.array([[0.1, 0.2, 0.3]])
    w2 = np.array([[0.7], [-0.8], [0.9]])
    b2 = np.array([[0.4]])
    
    mlp.layers[0].weights = Tensor(w1)
    mlp.layers[0].bias = Tensor(b1)
    mlp.layers[1].weights = Tensor(w2)
    mlp.layers[1].bias = Tensor(b2)

    x_data = np.array([[0.5, -0.5]])
    x_nano = Tensor(x_data)
    out_nano = mlp(x_nano)
    out_nano.backward()

    # PyTorch counterpart
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    w1_pt = torch.tensor(w1, requires_grad=True, dtype=torch.double)
    b1_pt = torch.tensor(b1, requires_grad=True, dtype=torch.double)
    w2_pt = torch.tensor(w2, requires_grad=True, dtype=torch.double)
    b2_pt = torch.tensor(b2, requires_grad=True, dtype=torch.double)

    h_pt = torch.relu(x_pt @ w1_pt + b1_pt)
    out_pt = torch.relu(h_pt @ w2_pt + b2_pt)
    out_pt.backward(torch.ones_like(out_pt))

    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(mlp.layers[0].weights.grad, w1_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(mlp.layers[0].bias.grad, b1_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(mlp.layers[1].weights.grad, w2_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(mlp.layers[1].bias.grad, b2_pt.grad.numpy(), rtol=1e-5)

def test_loss_and_optimizer():
    """Test Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) optimizer."""
    np.random.seed(42)
    
    # Model parameters
    w = Tensor(np.array([[1.5], [-2.0]]))
    b = Tensor(np.array([[0.5]]))
    
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true_data = np.array([[2.5], [5.5]])
    
    x = Tensor(x_data)
    y_true = Tensor(y_true_data)
    
    # Optimizer & Loss
    optimizer = SGD([w, b], learning_rate=0.01)
    criterion = MSE()
    
    # Forward pass
    y_pred = x @ w + b
    loss = criterion(y_pred, y_true)
    
    # Test zero_grad
    optimizer.zero_grad()
    assert np.all(w.grad == 0.0)
    assert np.all(b.grad == 0.0)
    
    loss.backward()
    
    # Keep nanograd gradients for comparison
    w_grad_nano = w.grad.copy()
    b_grad_nano = b.grad.copy()
    
    # Optimizer step
    w_old = w.data.copy()
    b_old = b.data.copy()
    optimizer.step()
    
    # Verify weight updates
    np.testing.assert_allclose(w.data, w_old - 0.01 * w_grad_nano)
    np.testing.assert_allclose(b.data, b_old - 0.01 * b_grad_nano)
    
    # Compare with PyTorch reference
    x_pt = torch.tensor(x_data, requires_grad=False, dtype=torch.double)
    y_true_pt = torch.tensor(y_true_data, requires_grad=False, dtype=torch.double)
    w_pt = torch.tensor([[1.5], [-2.0]], requires_grad=True, dtype=torch.double)
    b_pt = torch.tensor([[0.5]], requires_grad=True, dtype=torch.double)
    
    y_pred_pt = x_pt @ w_pt + b_pt
    loss_pt = torch.mean((y_pred_pt - y_true_pt) ** 2)
    loss_pt.backward()
    
    np.testing.assert_allclose(loss.data, loss_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(w_grad_nano, w_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b_grad_nano, b_pt.grad.numpy(), rtol=1e-5)

def test_softmax():
    """
    Tests the softmax activation function's forward and backward passes.
    Compares the calculated outputs and gradients against PyTorch's torch.softmax.
    """
    x_data = np.array([[1.0, 2.0, 3.0], 
                       [-1.0, 0.0, 1.0], 
                       [2.0, 2.0, 2.0]])
    weights_data = np.array([[0.2, 0.5, 0.3],
                             [0.1, 0.8, 0.1],
                             [0.4, 0.4, 0.2]])
    
    # 1. Forward and backward pass in nanograd
    x_nano = Tensor(x_data)
    out_nano = softmax(x_nano)
    # Perform a weighted sum to ensure non-zero gradients
    loss_nano = (out_nano * Tensor(weights_data)).sum()
    loss_nano.backward()

    # 2. Forward and backward pass in PyTorch
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    out_pt = torch.softmax(x_pt, dim=1)
    weights_pt = torch.tensor(weights_data, dtype=torch.double)
    loss_pt = (out_pt * weights_pt).sum()
    loss_pt.backward()

    # 3. Verify correctness
    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5, err_msg="Softmax forward values mismatch")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg="Softmax backward gradients mismatch")

def test_softmax_cross_entropy():
    """
    Tests the SoftmaxCrossEntropy loss function's forward and backward passes.
    Compares the loss values and gradients of logits against PyTorch's formulation.
    """
    # 3 samples, 3 classes
    logits_data = np.array([[1.5, -0.5, 2.0],
                            [0.0, 1.0, -1.0],
                            [-2.0, 2.0, 0.5]])
    
    # Soft target probability distributions (each row sums to 1.0)
    targets_data = np.array([[0.0, 0.0, 1.0],
                             [0.1, 0.8, 0.1],
                             [0.0, 1.0, 0.0]])
    
    # 1. Forward and backward pass in nanograd
    y_pred_nano = Tensor(logits_data, label='logits')
    y_true_nano = Tensor(targets_data, label='targets')
    criterion = SoftmaxCrossEntropy()
    loss_nano = criterion(y_pred_nano, y_true_nano)
    loss_nano.backward()
    
    # 2. Forward and backward pass in PyTorch using manual formulation of softmax CE with soft targets
    y_pred_pt = torch.tensor(logits_data, requires_grad=True, dtype=torch.double)
    y_true_pt = torch.tensor(targets_data, requires_grad=False, dtype=torch.double)
    
    log_probs = torch.log_softmax(y_pred_pt, dim=1)
    loss_pt = -(y_true_pt * log_probs).sum() / logits_data.shape[0]
    loss_pt.backward()
    
    # 3. Verify correctness
    np.testing.assert_allclose(loss_nano.data, loss_pt.detach().numpy(), rtol=1e-5, err_msg="SoftmaxCrossEntropy forward loss mismatch")
    np.testing.assert_allclose(y_pred_nano.grad, y_pred_pt.grad.numpy(), rtol=1e-5, err_msg="SoftmaxCrossEntropy backward gradients mismatch")

def test_conv2d():
    """
    Tests Conv2D forward and backward passes against PyTorch.
    """
    # Deterministic seed for reproducible testing
    np.random.seed(42)
    
    # 1. Initialize nanograd Conv2D layer
    in_channels = 3
    num_filters = 4
    kernel_size = (3, 3)
    padding = 1
    stride = 1
    
    conv_nano = Conv2D(in_channels, num_filters, kernel_size, padding, stride)
    
    # 2. Input data of shape (batch_size=2, channels=3, height=5, width=5)
    x_data = np.random.uniform(-1, 1, size=(2, in_channels, 5, 5))
    x_nano = Tensor(x_data)
    
    # Forward and backward pass in nanograd
    out_nano = conv_nano(x_nano)
    loss_nano = out_nano.sum()
    loss_nano.backward()
    
    # 3. Create PyTorch equivalent
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    
    # Reshape weights from (num_filters, in_channels, kh * kw) to (num_filters, in_channels, kh, kw)
    w_pt_data = conv_nano.weights.data.reshape(num_filters, in_channels, kernel_size[0], kernel_size[1])
    w_pt = torch.tensor(w_pt_data, requires_grad=True, dtype=torch.double)
    
    # Reshape bias from (1, num_filters) to (num_filters)
    b_pt_data = conv_nano.bias.data.reshape(num_filters)
    b_pt = torch.tensor(b_pt_data, requires_grad=True, dtype=torch.double)
    
    # Forward and backward pass in PyTorch
    out_pt = F.conv2d(x_pt, w_pt, b_pt, stride=stride, padding=padding)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    
    # 4. Assert correctness of forward and backward passes
    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5, err_msg="Conv2D forward pass mismatch")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg="Conv2D input gradient mismatch")
    np.testing.assert_allclose(conv_nano.bias.grad, b_pt.grad.numpy().reshape(1, num_filters), rtol=1e-5, err_msg="Conv2D bias gradient mismatch")
    np.testing.assert_allclose(conv_nano.weights.grad, w_pt.grad.numpy().reshape(num_filters, in_channels, -1), rtol=1e-5, err_msg="Conv2D weights gradient mismatch")

def test_max_pool2d():
    """
    Tests MaxPool2D forward and backward passes against PyTorch.
    """
    np.random.seed(42)
    
    # 1. Initialize MaxPool2D
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    pool_nano = MaxPool2D(kernel_size, stride, padding)
    
    # Input data of shape (batch_size=2, channels=3, height=4, width=4)
    # Ensure values are distinct to avoid argmax ambiguity issues
    x_data = np.random.uniform(-1, 1, size=(2, 3, 4, 4))
    x_nano = Tensor(x_data)
    
    # Forward and backward pass in nanograd
    out_nano = pool_nano(x_nano)
    loss_nano = out_nano.sum()
    loss_nano.backward()
    
    # 2. PyTorch equivalent
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    out_pt = F.max_pool2d(x_pt, kernel_size=kernel_size, stride=stride, padding=padding)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    
    # 3. Assert correctness of forward and backward passes
    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5, err_msg="MaxPool2D forward pass mismatch")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg="MaxPool2D input gradient mismatch")

def test_adam():
    """
    Tests the Adam optimizer's step updates against PyTorch.
    """
    np.random.seed(42)
    
    # Initialize deterministic weights and biases
    w_data = np.array([[1.5], [-2.0]])
    b_data = np.array([[0.5]])
    
    # 1. Nanograd setup
    w_nano = Tensor(w_data.copy())
    b_nano = Tensor(b_data.copy())
    optimizer_nano = Adam([w_nano, b_nano], learning_rate=0.1)
    
    # 2. PyTorch setup
    w_pt = torch.tensor(w_data.copy(), requires_grad=True, dtype=torch.double)
    b_pt = torch.tensor(b_data.copy(), requires_grad=True, dtype=torch.double)
    optimizer_pt = torch.optim.Adam([w_pt, b_pt], lr=0.1)
    
    # Run 3 steps of optimization
    for step in range(3):
        # Forward/backward in nanograd
        loss_nano = (w_nano ** 2).sum() + (b_nano ** 2).sum()
        optimizer_nano.zero_grad()
        loss_nano.backward()
        optimizer_nano.step()
        
        # Forward/backward in PyTorch
        loss_pt = (w_pt ** 2).sum() + (b_pt ** 2).sum()
        optimizer_pt.zero_grad()
        loss_pt.backward()
        optimizer_pt.step()
        
        # Compare weights and biases after each step
        np.testing.assert_allclose(w_nano.data, w_pt.detach().numpy(), rtol=1e-5, err_msg=f"Adam weight mismatch at step {step}")
        np.testing.assert_allclose(b_nano.data, b_pt.detach().numpy(), rtol=1e-5, err_msg=f"Adam bias mismatch at step {step}")

def test_flatten():
    """
    Tests the Flatten layer's forward and backward passes against PyTorch.
    """
    np.random.seed(42)
    
    # 1. Input data of shape (batch_size=2, channels=3, height=4, width=4)
    x_data = np.random.uniform(-1, 1, size=(2, 3, 4, 4))
    
    x_nano = Tensor(x_data)
    flatten_nano = Flatten()
    out_nano = flatten_nano(x_nano)
    loss_nano = out_nano.sum()
    loss_nano.backward()
    
    # 2. PyTorch equivalent
    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.double)
    out_pt = torch.flatten(x_pt, start_dim=1)
    loss_pt = out_pt.sum()
    loss_pt.backward()
    
    # 3. Assertions
    np.testing.assert_allclose(out_nano.data, out_pt.detach().numpy(), rtol=1e-5, err_msg="Flatten forward pass mismatch")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg="Flatten input gradient mismatch")



