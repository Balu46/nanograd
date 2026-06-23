import pytest
import numpy as np
import torch
from nanograd.tensor import Tensor
from nanograd.nn import relu, Layer, MLP
from nanograd.loss import MSE
from nanograd.optim import SGD

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
