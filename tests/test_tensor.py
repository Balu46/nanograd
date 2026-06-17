import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nanograd.tensor import Tensor

def test_tensor_initialization():
    # Inicjalizacja z listy
    t1 = Tensor([1.0, 2.0, 3.0])
    assert isinstance(t1.data, np.ndarray), "Dane powinny być trzymane jako np.ndarray"
    assert t1.data.tolist() == [1.0, 2.0, 3.0]
    
    # Inicjalizacja bezpośrednio z tablicy NumPy
    t2 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert t2.data.shape == (2, 2)
    
    # Inicjalizacja gradientu
    assert t2.grad is not None, "Gradient musi być zainicjalizowany"
    assert t2.grad.shape == t2.data.shape, "Gradient musi mieć ten sam wymiar co dane"
    assert np.all(t2.grad == 0), "Początkowy gradient powinien składać się z samych zer"