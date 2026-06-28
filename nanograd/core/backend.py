
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def get_xp(array):
    """
    Dyspozytor. Zwraca bibliotekę 'cupy', jeśli 'array' leży w VRAM, 
    lub 'numpy', jeśli 'array' leży w RAM.
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(array)
    return np



