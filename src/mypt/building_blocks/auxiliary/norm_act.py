"""
This module contains a block that applies 
"""

import abc
import torch

from torch import nn
from typing import Callable, List, OrderedDict, Union

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.auxiliary.normalization.utils import get_normalization
from mypt.building_blocks.auxiliary.activations.activations import get_activation
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin


class NormActBlock(WrapperLikeModuleMixin):
    """
    A block that applies a normalization and an activation function to the input.
    """

    def __init__(self, 
                 normalization: Union[str, Callable],
                 normalization_params: dict,
                 activation: Union[str, Callable],
                 activation_params: dict,
                 ):
        
        super().__init__("_block")  

        block_ordered_dict = OrderedDict({"normalization": get_normalization(normalization, normalization_params)})

        block_ordered_dict["activation"] = get_activation(activation, activation_params)

        self._block = nn.Sequential(block_ordered_dict)


    # the wrapper like module mixin overrides the main torch.nn.Module methods
    # ...

    @property
    def block(self) -> nn.Sequential:
        return self._block


class ConditionedNormActBlock(NonSequentialModuleMixin, abc.ABC):
    """
    A block that applies a normalization and an activation function to the input.
    """

    def __init__(self, 
                 normalization: Union[str, Callable],
                 normalization_params: dict,
                 activation: Union[str, Callable],
                 activation_params: dict,
                 inner_components_fields: List[str]
                ):
        super().__init__(inner_components_fields)

        self._normalization = get_normalization(normalization, normalization_params)
        self._activation = get_activation(activation, activation_params) 


    @abc.abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)