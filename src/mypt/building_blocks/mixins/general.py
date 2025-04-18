import torch
from torch import nn


class ModuleListMixin:
    """
    This mixin is used to provide default common implementations for a class that uses a ModuleList field.
    Most these methods are expected to be part of more thorough implementation in the child class.
    """
    def __init__(self, inner_model_field_name: str = '_block'):
        self._inner_model_field_name = inner_model_field_name

    def _verify_instance(self):
        if not hasattr(self, self._inner_model_field_name):
            raise AttributeError(f"the child class is expected to have the attribute '{self._inner_model_field_name}'")
        
        if not isinstance(getattr(self, self._inner_model_field_name), (torch.nn.ModuleList, nn.Sequential)):
            raise TypeError(f"The SequentialModuleListMixin expects the self._block attribute to be of type {torch.nn.ModuleList} or {nn.Sequential}. Found: {type(getattr(self, self._inner_model_field_name))}")

    def module_list_to(self, *args, **kwargs):
        self._verify_instance()

        inner_model = getattr(self, self._inner_model_field_name)

        # call the '.to' method for each Module in the ModuleList
        for i in range(len(inner_model)):
            inner_model[i] = inner_model[i].to(*args, **kwargs)
        
        # always return self
        return self 


class SequentialModuleListMixin(ModuleListMixin):
    """
    This mixin provides an implementation of the forward method for a class that uses a ModuleList but with a sequential structure.
    """

    def forward_sequential_module_list(self, x: torch.Tensor) -> torch.Tensor:
        self._verify_instance()

        inner_model = getattr(self, self._inner_model_field_name)

        # call the forward method for each Module in the ModuleList
        for i in range(len(inner_model)):
            x = inner_model[i](x)
        return x

