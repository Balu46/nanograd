import numpy as np
from nanograd.core.tensor import Tensor

class Parameter(Tensor):
    """
    A Parameter is a Tensor that is registered as a learnable parameter of a Module.
    """
    def __init__(self, data, label: str = ''):
        from nanograd.core.backend import get_xp
        xp = get_xp(data)
        if not isinstance(data, xp.ndarray):
            data = xp.array(data)
        super().__init__(data, label=label)


class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        
    def to(self, device: str):
        """
        Moves all parameters of this module and its sub-modules to the specified device.
        
        Args:
            device (str): The target device ('cpu' or 'cuda').
            
        Returns:
            Module: self, allowing for method chaining.
        """
        # 1. Move parameters belonging directly to this module
        for name, param in self._parameters.items():
            # The Tensor.to() method returns a new Tensor on the target device
            moved_tensor = param.to(device)
            
            # We must re-wrap it in a Parameter to maintain its identity and flags
            moved_param = Parameter(moved_tensor.data, label=param.label)
            if param.grad is not None:
                # Assuming grad is already moved by Tensor.to(), we just assign it
                moved_param.grad = moved_tensor.grad
                
            # Update the internal registry
            self._parameters[name] = moved_param
            # Update the class attribute (so self.W points to the new GPU parameter)
            object.__setattr__(self, name, moved_param)

        # 2. Recursively move all sub-modules (e.g., layers inside a Sequential container or list)
        for name, value in self.__dict__.items():
            if name in ('_modules', '_parameters'):
                continue
            if isinstance(value, Module):
                value.to(device)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.to(device)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        item.to(device)
            
        return self

    def cuda(self):
        """Shortcut to move all parameters to GPU."""
        return self.to('cuda')

    def cpu(self):
        """Shortcut to move all parameters to CPU."""
        return self.to('cpu')

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
        from nanograd.core.backend import get_xp
        for p in self.parameters():
            xp = get_xp(p.data)
            p.grad = xp.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
