import numpy as np
from nanograd.core.tensor import Tensor

class Parameter(Tensor):
    """
    A Parameter is a Tensor that is registered as a learnable parameter of a Module.
    """
    def __init__(self, data: np.ndarray, label: str = ''):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        super().__init__(data, label=label)


class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Tensor) and name in ('weights', 'bias', 'weight'):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> list:
        """
        Returns an iterator over module parameters, recursively collecting parameters from all submodules.
        """
        params = list(self._parameters.values())
        
        for name, value in self.__dict__.items():
            if name == '_modules':
                continue
            if isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                        
        seen = set()
        unique_params = []
        for p in params:
            if p not in seen:
                seen.add(p)
                unique_params.append(p)
        return unique_params

    def params(self) -> list:
        """Alias for parameters() to support existing MLP usage."""
        return self.parameters()

    def model_params(self) -> list:
        """Alias for parameters() to support existing Layer usage."""
        return self.parameters()

    def zero_grad(self):
        """Resets gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
