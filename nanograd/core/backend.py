
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def get_xp(array):
    """
    Dispatcher. Returns 'cupy' module if the input array resides in GPU VRAM,
    or 'numpy' if it resides in system RAM.
    """
    if CUPY_AVAILABLE:
        return cp.get_array_module(array)
    return np



