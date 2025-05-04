from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from typing import Iterator, Optional, Set, Tuple

class WrapperLikeModuleMixin(nn.Module):
    """
    This Mixin is used to override the most important methods of a the nn.Module class when creating a custom module.
    Methods such as children(), named_children(), modules(), named_modules(), parameters(), named_parameters(), etc. are overridden to return the appropriate values.
    """
    def __init__(self, inner_model_field_name: str):
        super().__init__()
        self._inner_model_field_name = inner_model_field_name

    def _verify_instance_wrapperLikeModuleMixin(self):
        if not hasattr(self, self._inner_model_field_name):
            raise AttributeError(f"the child class is expected to have the attribute '{self._inner_model_field_name}'")

        if not isinstance(getattr(self, self._inner_model_field_name), nn.Module):
            raise TypeError(f"The WrapperLikeModuleMixin expects the self._block attribute to be of type {nn.Module}. Found: {type(getattr(self, self._inner_model_field_name))}")

    # the children-related methods
    def children(self) -> Iterator[nn.Module]:
        return getattr(self, self._inner_model_field_name).children()

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return getattr(self, self._inner_model_field_name).named_children()

    # the modules-related methods
    def modules(self) -> Iterator[nn.Module]:
        return getattr(self, self._inner_model_field_name).modules()

    # !!!! MAKE SURE TO PRESERVE THE METHOD SIGNATURE WHEN OVERRIDING 
    # THE METHODS OF A PARENT CLASS
    def named_modules(self, 
                    memo: Optional[Set[nn.Module]] = None,
                    prefix: str = "",
                    remove_duplicate: bool = True,
                      ) -> Iterator[Tuple[str, nn.Module]]:
        return getattr(self, self._inner_model_field_name).named_modules(memo, prefix, remove_duplicate)

    # the parameters-related methods    
    def parameters(self) -> Iterator[nn.Parameter]:
        return getattr(self, self._inner_model_field_name).parameters()

    # !!!! MAKE SURE TO PRESERVE THE METHOD SIGNATURE WHEN OVERRIDING 
    # THE METHODS OF A PARENT CLASS
    def named_parameters(self, 
                        prefix: str = "", 
                        recurse: bool = True, 
                        remove_duplicate: bool = True
                        ) -> Iterator[Tuple[str, nn.Parameter]]:   
        return getattr(self, self._inner_model_field_name).named_parameters(prefix, recurse, remove_duplicate)

    # the states-related methods
    def train(self, mode: bool = True) -> 'WrapperLikeModuleMixin':
        self.training = mode
        # assign the wrapped module to its new state
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).train(mode)) 
        # and then return the entire object
        return self

    def eval(self) -> 'WrapperLikeModuleMixin':
        self.training = False
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).eval())
        return self
    
    def to(self, *args, **kwargs) -> 'WrapperLikeModuleMixin':
        setattr(self, self._inner_model_field_name, getattr(self, self._inner_model_field_name).to(*args, **kwargs))
        return self

    # the forward method    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return getattr(self, self._inner_model_field_name).forward(x, *args, **kwargs) 
    
    def __str__(self) -> str:
        return getattr(self, self._inner_model_field_name).__str__() 
    
    def __repr__(self) -> str:
        return getattr(self, self._inner_model_field_name).__repr__() 
    


class CloneableModuleMixin(ABC):
    """
    This Mixin is used to override the __call__ method of a module to return a clone of the module.
    """
    @abstractmethod
    def get_constructor_args(self) -> dict:
        """
        This method should return a dictionary of the arguments that should be passed to the constructor of the module.
        """
        pass

    def _verify_instance_cloneableModuleMixin(self):
        if not isinstance(self, nn.Module):
            raise TypeError(f"the child class is expected to be a subclass of {nn.Module}")

    def clone(self) -> 'CloneableModuleMixin':
        self._verify_instance_cloneableModuleMixin()
        # get the constructor arguments with their values
        constructor_args = self.get_constructor_args() 

        # define the module
        module = self.__class__(**constructor_args) 
        # we know that `self` represents a class that extends `torch.nn.Module`
        # and hence does `module`
        # load the state dict
        module.load_state_dict(self.state_dict())

        return module

