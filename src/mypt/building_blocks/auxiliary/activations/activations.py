"""
This module provides a way to work uniformly with different activation functions.
It will be extended after exploring the diffusers library (this file is inspired by the diffusers library)
"""


from torch import nn
from typing import Callable, Optional, Union

_activation_functions = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink,
    "silu": nn.SiLU,
}


def get_activation(activation: Union[str, Callable], activation_params: Optional[dict] = None) -> nn.Module:
    """
    Get an activation function from a string or a callable.
    """
    # make sure to set the activation_params to an empty dict if it is None 
    if activation_params is None:
        activation_params = {}

    if isinstance(activation, str):
        activation = activation.lower()
    
        if activation in _activation_functions:
            return _activation_functions[activation](**activation_params)
        
        raise ValueError(f"Unsupported activation function: {activation}")


    elif isinstance(activation, Callable):
        return activation(**activation_params)

    raise TypeError(f"Activation must be a string or a callable, got {type(activation)}")
