from torch import nn
from typing import Callable, Optional, Union


_normalization_functions = {
    # batch normalization
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,

    # TODO: UNDERSTAND INSTANCE NORMALIZATION AND ADD IT TO THE TESTS 

    # instance normalization
    # "instancenorm1d": nn.InstanceNorm1d,
    # "instancenorm2d": nn.InstanceNorm2d,


    # layer normalization
    # TODO: UNDERSTAND LAYERNORM AND ADD IT TO THE TESTS
    # "layernorm": nn.LayerNorm,    

    # group normalization
    "groupnorm": nn.GroupNorm,
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
        return normalization(**normalization_params)
    
    raise TypeError(f"Normalization must be a string or a callable, got {type(normalization)}")

