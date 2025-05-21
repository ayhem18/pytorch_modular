from torch import nn
from typing import Callable, Optional, Union


_normalization_functions = {
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "layernorm": nn.LayerNorm,
    "instancenorm": nn.InstanceNorm2d,
    "groupnorm": nn.GroupNorm,
    "groupnorm2d": nn.GroupNorm2d,
}


def get_normalization(normalization: Union[str, Callable], normalization_params: Optional[dict] = None) -> nn.Module:
    """
    Get a normalization function from a string or a callable.
    """ 
    
    if isinstance(normalization, str):
        normalization = normalization.lower()
    
        if normalization in _normalization_functions:
            return _normalization_functions[normalization](**normalization_params)
    
        raise ValueError(f"Unsupported normalization function: {normalization}")

    elif isinstance(normalization, Callable):
        return normalization(normalization_params)
    
    raise TypeError(f"Normalization must be a string or a callable, got {type(normalization)}")

