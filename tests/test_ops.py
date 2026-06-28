import pytest
import numpy as np
import torch
from nanograd import Tensor

# =====================================================================
# 1. BINARY OPERATION TESTS (Two variables, e.g., addition, multiplication)
# =====================================================================

@pytest.mark.parametrize("op_name, op_nano, op_torch", [
    ("addition", lambda a, b: a + b, lambda a, b: a + b),
    ("subtraction", lambda a, b: a - b, lambda a, b: a - b),
    ("multiplication", lambda a, b: a * b, lambda a, b: a * b),
    ("division", lambda a, b: a / b, lambda a, b: a / b),
    # Power is tested with positive base values to avoid complex numbers for fractional exponents
    ("power", lambda a, b: a ** b, lambda a, b: a ** b),
])
def test_binary_operations(op_name, op_nano, op_torch):
    """Tests all basic binary operations on two Tensors."""
    
    # Positive values are used due to power operation constraints
    data_a = [[2.0, 3.0], [4.0, 5.0]]
    data_b = [[1.5, 2.0], [0.5, 3.0]]

    # --- nanograd ---
    a_nano = Tensor(np.array(data_a), label='a')
    b_nano = Tensor(np.array(data_b), label='b')
    res_nano = op_nano(a_nano, b_nano)
    res_nano.backward()

    # --- PyTorch ---
    a_pt = torch.tensor(data_a, requires_grad=True)
    b_pt = torch.tensor(data_b, requires_grad=True)
    res_pt = op_torch(a_pt, b_pt)
    res_pt.backward(torch.ones_like(res_pt))

    # --- Assertions (with rtol=1e-5 tolerance for float rounding differences) ---
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5, err_msg=f"Value error (forward) for: {op_name}")
    np.testing.assert_allclose(a_nano.grad, a_pt.grad.numpy(), rtol=1e-5, err_msg=f"Gradient error 'a' for: {op_name}")
    np.testing.assert_allclose(b_nano.grad, b_pt.grad.numpy(), rtol=1e-5, err_msg=f"Gradient error 'b' for: {op_name}")


# =====================================================================
# 2. UNARY OPERATION TESTS (Single variable, e.g., exp, negation)
# =====================================================================

@pytest.mark.parametrize("op_name, op_nano, op_torch", [
    ("negation (-x)", lambda x: -x, lambda x: -x),
    ("exp", lambda x: x.exp(), lambda x: torch.exp(x)),
    ("sum", lambda x: x.sum(), lambda x: torch.sum(x)),
    ("max", lambda x: x.max(), lambda x: torch.max(x)),
])
def test_unary_operations(op_name, op_nano, op_torch):
    """Tests unary operations performed on a single Tensor."""
    
    data = [[-1.0, 2.0], [0.5, -0.5]]

    # --- nanograd ---
    x_nano = Tensor(np.array(data), label='x')
    res_nano = op_nano(x_nano)
    res_nano.backward()

    # --- PyTorch ---
    x_pt = torch.tensor(data, requires_grad=True)
    res_pt = op_torch(x_pt)
    res_pt.backward(torch.ones_like(res_pt))

    # --- Assertions ---
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5, err_msg=f"Value error (forward) for: {op_name}")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg=f"Gradient error for: {op_name}")


# =====================================================================
# 3. COMPLEX EXPRESSION TEST (Chain Rule verification)
# =====================================================================

def test_complex_chain_rule():
    """Tests a complex mathematical expression to verify the Chain Rule."""
    
    data_x = [[1.0, -2.0], [3.0, 0.5]]
    data_y = [[0.5, 1.5], [-1.0, 2.0]]

    # --- nanograd ---
    x_nano = Tensor(np.array(data_x), label='x')
    y_nano = Tensor(np.array(data_y), label='y')
    
    # Expression: z = sum( exp(x * y) - (x / y) )
    z_nano = ((x_nano * y_nano).exp() - (x_nano / y_nano)).sum()
    z_nano.backward()

    # --- PyTorch ---
    x_pt = torch.tensor(data_x, requires_grad=True)
    y_pt = torch.tensor(data_y, requires_grad=True)
    
    z_pt = torch.sum(torch.exp(x_pt * y_pt) - (x_pt / y_pt))
    z_pt.backward(torch.ones_like(z_pt))

    # --- Assertions ---
    np.testing.assert_allclose(z_nano.data, z_pt.detach().numpy(), rtol=1e-5, err_msg="Value error in complex test")
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5, err_msg="Gradient error 'x' in complex test")
    np.testing.assert_allclose(y_nano.grad, y_pt.grad.numpy(), rtol=1e-5, err_msg="Gradient error 'y' in complex test")


# =====================================================================
# 4. MATRIX MULTIPLICATION TEST (@)
# =====================================================================

def test_matmul():
    """Tests matrix multiplication (@) and its backward propagation."""
    data_a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2x3
    data_b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  # 3x2

    # --- nanograd ---
    a_nano = Tensor(np.array(data_a))
    b_nano = Tensor(np.array(data_b))
    res_nano = a_nano @ b_nano
    res_nano.backward()

    # --- PyTorch ---
    a_pt = torch.tensor(data_a, requires_grad=True, dtype=torch.double)
    b_pt = torch.tensor(data_b, requires_grad=True, dtype=torch.double)
    res_pt = a_pt @ b_pt
    res_pt.backward(torch.ones_like(res_pt))

    # --- Assertions ---
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(a_nano.grad, a_pt.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b_nano.grad, b_pt.grad.numpy(), rtol=1e-5)


# =====================================================================
# 5. SCALAR OPERATION TESTS (Left and right side)
# =====================================================================

def test_scalar_operations():
    """Tests mathematical operations between Tensors and python numbers (scalars)."""
    data_x = [[2.0, 3.0], [4.0, 5.0]]

    # 1. Adding a scalar from the right side (x + 2.0)
    x_nano = Tensor(np.array(data_x))
    res_nano = x_nano + 2.0
    res_nano.backward()
    
    x_pt = torch.tensor(data_x, requires_grad=True, dtype=torch.double)
    res_pt = x_pt + 2.0
    res_pt.backward(torch.ones_like(res_pt))
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5)

    # 2. Multiplying a scalar from the left side (3.0 * x) - testing __rmul__
    x_nano = Tensor(np.array(data_x))
    res_nano = 3.0 * x_nano
    res_nano.backward()
    
    x_pt = torch.tensor(data_x, requires_grad=True, dtype=torch.double)
    res_pt = 3.0 * x_pt
    res_pt.backward(torch.ones_like(res_pt))
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5)

    # 3. Subtracting a scalar from the left side (5.0 - x) - testing __rsub__
    x_nano = Tensor(np.array(data_x))
    res_nano = 5.0 - x_nano
    res_nano.backward()
    
    x_pt = torch.tensor(data_x, requires_grad=True, dtype=torch.double)
    res_pt = 5.0 - x_pt
    res_pt.backward(torch.ones_like(res_pt))
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5)

    # 4. Dividing by a scalar from the right side (x / 2.0)
    x_nano = Tensor(np.array(data_x))
    res_nano = x_nano / 2.0
    res_nano.backward()
    
    x_pt = torch.tensor(data_x, requires_grad=True, dtype=torch.double)
    res_pt = x_pt / 2.0
    res_pt.backward(torch.ones_like(res_pt))
    np.testing.assert_allclose(res_nano.data, res_pt.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(x_nano.grad, x_pt.grad.numpy(), rtol=1e-5)