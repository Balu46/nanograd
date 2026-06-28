import os
import sys
import numpy as np
import pytest
import torch

from nanograd import Tensor

def test_autograd_complex_expression(device, to_np):
    """
    Tests the expression: e = c * (a + b) and compares the gradients with PyTorch.
    """
    # 1. PREPARE DATA
    a_data = [[1.0, 2.0], [3.0, 4.0]]
    b_data = [[5.0, 6.0], [7.0, 8.0]]
    c_data = [[1.0, 2.0], [1.0, 1.0]]

    # 2. COMPUTATIONS IN YOUR ENGINE (nanograd)
    a_nano = Tensor(np.array(a_data), label='a').to(device)
    b_nano = Tensor(np.array(b_data), label='b').to(device)
    c_nano = Tensor(np.array(c_data), label='c').to(device)

    e_nano = c_nano * (a_nano + b_nano)
    e_nano.backward()

    # 3. COMPUTATIONS IN THE REFERENCE ENGINE (PyTorch)
    a_pt = torch.tensor(a_data, requires_grad=True)
    b_pt = torch.tensor(b_data, requires_grad=True)
    c_pt = torch.tensor(c_data, requires_grad=True)

    e_pt = c_pt * (a_pt + b_pt)
    e_pt.backward(torch.ones_like(e_pt))

    # 4. ASSERTIONS (VERIFY RESULTS)
    # Check if the output values (forward pass) are equal
    np.testing.assert_allclose(to_np(e_nano), e_pt.detach().numpy(), err_msg="Error in forward pass")

    # Check if the gradients (backward pass) are equal
    np.testing.assert_allclose(to_np(a_nano.grad), a_pt.grad.numpy(), err_msg="Incorrect gradient for 'a'")
    np.testing.assert_allclose(to_np(b_nano.grad), b_pt.grad.numpy(), err_msg="Incorrect gradient for 'b'")
    np.testing.assert_allclose(to_np(c_nano.grad), c_pt.grad.numpy(), err_msg="Incorrect gradient for 'c'")

def test_simple_addition(device, to_np):
    """A smaller test verifying addition only."""
    a = Tensor(np.array([1.0, 2.0])).to(device)
    b = Tensor(np.array([3.0, 4.0])).to(device)
    c = a + b
    c.backward()
    
    # The derivative of addition is always 1 for each element
    np.testing.assert_allclose(to_np(a.grad), np.array([1.0, 1.0]))
    np.testing.assert_allclose(to_np(b.grad), np.array([1.0, 1.0]))