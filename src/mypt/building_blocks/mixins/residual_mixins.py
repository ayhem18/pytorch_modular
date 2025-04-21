import torch

from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from typing import Optional, Iterator, Union, Tuple


class GeneralResidualMixin:
    """
    This mixin is a general implementation of a residual connection.
    The idea is simple: have a main stream and a residual stream passing the same input x through different paths (whatever these paths are)
    and then add the outputs together. 

    This general scheme applies to: 
    
    - very first resnet residual blocks where the input x and the output are of the same shape and hence the skip 
    connection is basically an identity function 
    
    - a linear skip connection where a linear layer is used to project the input x to the same dimension of the output of the main stream. 
    """

    def __init__(self, main_stream_field_name: str, residual_stream_field_name: Optional[str] = None):
        self._main_stream_field_name = main_stream_field_name
        self._residual_stream_field_name = residual_stream_field_name



    def _get_main_stream(self) -> Union[nn.Module, nn.Sequential]:
        """Get the main stream component"""
        if not hasattr(self, self._main_stream_field_name):
            raise AttributeError(f"The class {self.__class__.__name__} expects the attribute '{self._main_stream_field_name}' to be set")

        if not isinstance(getattr(self, self._main_stream_field_name), nn.Module):
            raise TypeError(f"The attribute '{self._main_stream_field_name}' must be an instance of {nn.Module}")

        return getattr(self, self._main_stream_field_name)
    
    def _get_residual_stream(self) -> Optional[Union[nn.Module, nn.Sequential]]:
        """Get the residual stream component"""

        if self._residual_stream_field_name is None:
            return None

        if not hasattr(self, self._residual_stream_field_name):
            raise AttributeError(f"The class {self.__class__.__name__} expects the attribute '{self._residual_stream_field_name}' to be set")

        if not isinstance(getattr(self, self._residual_stream_field_name), nn.Module):
            raise TypeError(f"The attribute '{self._residual_stream_field_name}' must be an instance of {nn.Module}")

        return getattr(self, self._residual_stream_field_name)


    def residual_forward(self, x: torch.Tensor, debug:bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # first get the main stream output
        if debug:
            ms = self._get_main_stream() 
            
        main_stream_output = self._get_main_stream().forward(x)

        # if the residual stream is not set, then we can assume it is an identity function 
        if self._residual_stream_field_name is None:
            if main_stream_output.shape != x.shape:
                raise ValueError(f"The main stream output shape {main_stream_output.shape} does not match the input shape {x.shape}")

            if debug:
                return main_stream_output, x, main_stream_output + x 

            return main_stream_output + x 
        
        # otherwise, get the residual stream output
        residual_stream_output = self._get_residual_stream().forward(x)

        if residual_stream_output.shape != main_stream_output.shape:
            raise ValueError(f"The residual stream output shape {residual_stream_output.shape} does not match the main stream output shape {main_stream_output.shape}")
        
        if debug:
            return main_stream_output, residual_stream_output, main_stream_output + residual_stream_output

        return main_stream_output + residual_stream_output
        
    
    def residual_to(self, *args, **kwargs):
        # set the main stream to the new device
        setattr(self, self._main_stream_field_name, getattr(self, self._main_stream_field_name).to(*args, **kwargs))

        # if the residual stream is set, then set it to the new device as well
        if self._residual_stream_field_name is not None:
            setattr(self, self._residual_stream_field_name, getattr(self, self._residual_stream_field_name).to(*args, **kwargs))

        return self
        

    def residual_children(self) -> Iterator[Module]:
        """
        Returns an iterator over immediate children modules.
        Includes both main stream and residual stream modules.
        """
        # Yield all children from the main stream
        main_stream = self._get_main_stream()
        yield from main_stream.children()
                    
        # Yield all children from the residual stream
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            yield from residual_stream.children()
    
    
    def residual_named_children(self) -> Iterator[Tuple[str, Module]]:
        """
        Returns an iterator over immediate children modules, yielding both the name and the module.
        Includes both main stream and residual stream modules.
        """
        # Yield named children from the main stream
        main_stream = self._get_main_stream()
        yield from main_stream.named_children()
            
        # Yield named children from the residual stream
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            yield from residual_stream.named_children()
    
    
    def residual_modules(self) -> Iterator[Module]:
        """
        Returns an iterator over all modules in the network, recursively.
        Includes both main stream and residual stream modules.
        """
        # TODO: understand exactly how torch.nn.Module.modules() work 
        # Yield self first
        # yield self
        
        # Yield modules from the main stream
        main_stream = self._get_main_stream()
        yield from main_stream.modules()
        
        # Yield modules from the residual stream
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            yield from residual_stream.modules()
    

    def residual_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        Includes parameters from both main stream and residual stream.
        
        Args:
            recurse (bool): If True, returns parameters of this module and
                all submodules. Otherwise, returns only parameters that are
                direct members of this module.
        """
        # Get parameters from the main stream
        main_stream = self._get_main_stream()
        yield from main_stream.parameters(recurse=recurse)
        
        # Get parameters from the residual stream
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            yield from residual_stream.parameters(recurse=recurse)
    

    def residual_named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the name and the parameter.
        Includes parameters from both main stream and residual stream.
        
        Args:
            prefix (str): Prefix to prepend to parameter names
            recurse (bool): If True, returns parameters of this module and
                all submodules. Otherwise, returns only parameters that are
                direct members of this module.
        """
        # Get named parameters from the main stream
        main_stream = self._get_main_stream()
        main_prefix = f"{prefix}.{self._main_stream_field_name}" if prefix else self._main_stream_field_name
        yield from main_stream.named_parameters(prefix=main_prefix, recurse=recurse)
        
        # Get named parameters from the residual stream
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            residual_prefix = f"{prefix}.{self._residual_stream_field_name}" if prefix else self._residual_stream_field_name
            yield from residual_stream.named_parameters(prefix=residual_prefix, recurse=recurse)
    
    
    def residual_to(self, *args, **kwargs) -> 'GeneralResidualMixin':
        """
        Moves and/or casts the parameters and buffers.
        Applies to both main stream and residual stream.
        """
        # Move/cast the main stream
        main_stream = self._get_main_stream()
        main_stream_to = main_stream.to(*args, **kwargs)
        setattr(self, self._main_stream_field_name, main_stream_to)
        
        # Move/cast the residual stream
        residual_stream = self._get_residual_stream()
        
        if residual_stream is not None:
            setattr(self, self._residual_stream_field_name, residual_stream.to(*args, **kwargs))
        
        return self
    

    def residual_train(self, mode: bool = True) -> 'GeneralResidualMixin':
        """
        Sets the module in training mode.
        Applies to both main stream and residual stream.
        
        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False)
        """
        # Set main stream to train mode
        main_stream = self._get_main_stream()
        main_stream.train(mode)
        
        # Set residual stream to train mode
        residual_stream = self._get_residual_stream()
        if residual_stream is not None:
            residual_stream.train(mode)
        
        return self
    
    
    def residual_eval(self) -> 'GeneralResidualMixin':
        """
        Sets the module in evaluation mode.
        Applies to both main stream and residual stream.
        """
        return self.residual_train(False)
        

