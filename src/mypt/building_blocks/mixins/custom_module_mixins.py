from typing import Iterator, Tuple
import torch.nn as nn


class WrapperLikeModuleMixin(nn.Module):
    """
    This Mixin is used to override the most important methods of a the nn.Module class when creating a custom module.
    Methods such as children(), named_children(), modules(), named_modules(), parameters(), named_parameters(), etc. are overridden to return the appropriate values.
    """
    def __init__(self, inner_model_field_name: str = '_block'):
        super().__init__()
        self._inner_model_field_name = inner_model_field_name

    def _verify_instance(self):
        if not hasattr(self, self._inner_model_field_name):
            raise AttributeError(f"the child class is expected to have the attribute '{self._inner_model_field_name}'")

        if not isinstance(getattr(self, self._inner_model_field_name), nn.Module):
            raise TypeError(f"The CustomModuleMixin expects the self._block attribute to be of type {nn.Module}. Found: {type(getattr(self, self._inner_model_field_name))}")

    # the children-related methods
    def children(self) -> Iterator[nn.Module]:
        return getattr(self, self._inner_model_field_name).children()

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return getattr(self, self._inner_model_field_name).named_children()

    # the modules-related methods
    def modules(self) -> Iterator[nn.Module]:
        return getattr(self, self._inner_model_field_name).modules()

    def named_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        return getattr(self, self._inner_model_field_name).named_modules()

    # the parameters-related methods    
    def parameters(self) -> Iterator[nn.Parameter]:
        return getattr(self, self._inner_model_field_name).parameters()

    def named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return getattr(self, self._inner_model_field_name).named_parameters()

    # the states related methods
    def train(self, mode: bool = True) -> 'WrapperLikeModuleMixin':
        # assign the wrapped module to its new state
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).train(mode)) 
        # and then return the entire object
        return self

    def eval(self) -> 'WrapperLikeModuleMixin':
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).eval())
        return self
    
    def to(self, *args, **kwargs) -> 'WrapperLikeModuleMixin':
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).to(*args, **kwargs))
        return self
