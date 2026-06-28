# Nanograd

**Nanograd** is a lightweight, educational autograd (automatic differentiation) engine written in Python. It supports vector and matrix operations using `NumPy` under the hood and provides an API heavily inspired by PyTorch.

The project contains a complete implementation of backpropagation on computation graphs, along with basic neural network layers, loss functions, and optimizers.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Basic Autograd](#1-basic-autograd)
  - [2. Building and Training a Neural Network (MLP)](#2-building-and-training-a-neural-network-mlp)
- [Tutorials & Jupyter Notebooks](#tutorials--jupyter-notebooks)
- [Testing](#testing)
- [How It Works](#how-it-works)
- [License](#license)

---

## Features

*   **`Tensor` Class**: A wrapper around `numpy.ndarray` that tracks mathematical operations to build a computation graph.
*   **Automatic Differentiation**: Backpropagation through the directed acyclic graph (DAG) via the `.backward()` method, using topological sorting.
*   **Mathematical Operators**:
    *   Addition (`+`) and subtraction (`-`)
    *   Element-wise multiplication (`*`) and division (`/`)
    *   Matrix multiplication (`@`)
    *   Power (`**`) and exponential (`.exp()`)
    *   Summation over all elements (`.sum()`)
*   **Neural Network Components**:
    *   `relu` activation function (lowercase as a standard function)
    *   Fully-connected `Layer` class with random weight and bias initialization
    *   Multi-Layer Perceptron (`MLP`)
*   **Optimizer**: Stochastic Gradient Descent (`SGD`) supporting gradient resetting (`zero_grad()`) and parameter updates (`step()`).
*   **Loss Function**: Mean Squared Error (`MSE`).

---

## Project Structure

```text
Autograd/
│
├── nanograd/                 # Main package
│   ├── __init__.py           # Public API exports
│   ├── tensor.py             # Tensor class and core autograd operations
│   ├── nn.py                 # Neural network layers (Layer, MLP, relu, Conv2D, etc.)
│   ├── loss.py               # Loss functions (MSE, SoftmaxCrossEntropy)
│   └── optim.py              # Optimizers (SGD, Adam)
│
├── examples/                 # Interactive Jupyter Notebook tutorials
│   ├── nanograd_tutorial.ipynb       # Basic Tensors & MLP binary classification
│   ├── mnist_cnn_tutorial.ipynb      # Recreating LeNet-5 CNN on MNIST (96% acc)
│   ├── polynomial_regression.ipynb   # Fitting polynomial to sine curve via raw Tensors
│   ├── multiclass_spirals.ipynb      # 3-class spiral classification with softmax
│   └── loss_landscape_optimization.ipynb # Visualizing SGD vs Adam paths on Beale's surface
│
├── tests/                    # Unit tests comparing results with PyTorch
│   ├── __init__.py
│   ├── test_tensor.py
│   ├── test_ops.py
│   └── test_nn.py
│
├── pyproject.toml            # Pip package configuration and dependencies
└── README.md                 # Project documentation
```

---

## Installation

You can install the library locally using `pip` from the root directory of the project:

### Standard Installation
```bash
pip install .
```

### Editable Installation (Development Mode)
Recommended if you plan to modify the source code:
```bash
pip install -e .
```

### Installation with Test Dependencies (pytest, PyTorch for verification)
```bash
pip install -e ".[dev]"
```

### Install directly from GitHub
You can install the package directly from the GitHub repository using pip:
```bash
pip install git+https://github.com/Balu46/nanograd.git
```

---

## Quick Start

Once installed, you can import the library as `nanograd`:

```python
from nanograd import Tensor, MLP, SGD, MSE, relu
```

### 1. Basic Autograd

The following example demonstrates how to define Tensors, perform operations, and compute gradients:

```python
import numpy as np
from nanograd import Tensor

# Initialize Tensors (wrapping numpy arrays)
a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), label='a')
b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), label='b')
c = Tensor(np.array([[1.0, 2.0], [1.0, 1.0]]), label='c')

# Complex expression: e = c * (a + b)
e = c * (a + b)

# Perform backward pass (Backpropagation)
e.backward()

# Check values and gradients
print("Forward Result:\n", e.data)
print("Gradient of 'a':\n", a.grad)
print("Gradient of 'b':\n", b.grad)
```

---

### 2. Building and Training a Neural Network (MLP)

In this example, we build a simple MLP with 2 inputs, a hidden layer of 3 neurons, and 1 output. We train it on a toy dataset using the `MSE` loss and `SGD` optimizer.

```python
import numpy as np
from nanograd import Tensor, MLP, SGD, MSE

# 1. Prepare training data (e.g. XOR problem)
X_data = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y_data = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

X = Tensor(X_data)
y_true = Tensor(y_data)

# 2. Define the MLP model (input: 2, hidden: 3, output: 1)
model = MLP([2, 3, 1])

# 3. Define the optimizer and loss function
optimizer = SGD(model.params(), learning_rate=0.05)
criterion = MSE()

# 4. Training loop (100 epochs)
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    
    # Calculate loss
    loss = criterion(y_pred, y_true)
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update model parameters
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/100 | MSE Loss: {loss.data:.6f}")

print("\nPredictions after training:")
print(model(X).data)
```

---

## Tutorials & Jupyter Notebooks

We provide six interactive Jupyter Notebooks in the `examples/` directory to help you learn and explore the capabilities of the `nanograd` engine:

1.  **[nanograd_tutorial.ipynb](examples/nanograd_tutorial.ipynb) (Basic Tutorial)**:
    *   Learn how to build computation graphs and calculate gradients.
    *   Train an MLP to solve a non-linear concentric circles classification problem.
    *   Perform forward and backward passes using CNN components.
2.  **[mnist_cnn_tutorial.ipynb](examples/mnist_cnn_tutorial.ipynb) (MNIST LeNet-5)**:
    *   Recreate the classic **LeNet-5** CNN architecture.
    *   Train the model on the MNIST digits dataset using Adam and Softmax Cross Entropy.
    *   Achieve **96%+ test accuracy** and visualize predictions.
3.  **[polynomial_regression.ipynb](examples/polynomial_regression.ipynb) (Polynomial Regression)**:
    *   Fit a 3rd-degree polynomial to a noisy sine curve using raw Tensors.
    *   Manually code a gradient descent training loop without using layers.
4.  **[multiclass_spirals.ipynb](examples/multiclass_spirals.ipynb) (Multiclass Spirals)**:
    *   Classify the 3-class spiral dataset.
    *   Evaluate output boundaries with an MLP, Softmax, and Cross Entropy.
5.  **[loss_landscape_optimization.ipynb](examples/loss_landscape_optimization.ipynb) (Optimizer Trajectories)**:
    *   Trace the optimization path of SGD vs Adam on Beale's plateau function.
    *   Observe how Adam dynamically adjusts step size to navigate sharp valleys.

### How to Run the Notebooks:
First, install the library and Jupyter notebook dependencies:
```bash
pip install -e ".[dev]" notebook matplotlib
```
Then, launch the notebook server:
```bash
jupyter notebook examples/
```

---

## Testing

The project includes an extensive test suite that compares computed values and gradients against **PyTorch**.

To run the tests, first make sure you have the development dependencies installed:
```bash
pip install -e ".[dev]"
```

Then, run `pytest` in the root directory:
```bash
pytest
```

The test suite covers:
1.  **`tests/test_tensor.py`**: Basic arithmetic operations and complex graphs.
2.  **`tests/test_ops.py`**: Parameterized tests of binary operations (`+`, `-`, `*`, `/`, `**`) and unary operations (`negation`, `exp`, `sum`), as well as matrix multiplication and scalar operations.
3.  **`tests/test_nn.py`**: Tests for the `relu` activation, `Layer`, `MLP`, `MSE` loss, and `SGD` optimizer.

---

## How It Works

The core of the library is a directed acyclic graph (DAG) automatically constructed in the background during mathematical operations:

1.  Each `Tensor` stores references to its "parent" tensors in `self._prev` along with the operation type `self._op`.
2.  During the forward pass, a local closure function `_backward()` is created. This function computes local partial derivatives with respect to the parents and adds them (accumulates them) to their gradients (`self.grad`).
3.  Calling `.backward()` on the output tensor sorts the graph topologically using `topo_sort()`. This ensures that a node is processed only after all its dependent child nodes have been evaluated. It then iterates backwards through the sorted nodes, triggering the `_backward()` function for each node.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
