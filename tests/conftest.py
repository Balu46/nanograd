import pytest
import numpy as np
from nanograd import Tensor
from nanograd.core.backend import get_xp, CUPY_AVAILABLE

DEVICES = ['cpu']
if CUPY_AVAILABLE:
    import cupy as cp
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            DEVICES.append('cuda')
    except Exception:
        pass

@pytest.fixture(params=DEVICES)
def device(request):
    return request.param

@pytest.fixture
def to_np():
    def _to_np(x):
        if isinstance(x, Tensor):
            x = x.data
        if hasattr(x, 'get'):
            return x.get()
        return np.asarray(x)
    return _to_np
