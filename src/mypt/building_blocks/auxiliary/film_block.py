import torch

from torch import nn
from typing import Callable, Union

from mypt.building_blocks.auxiliary.norm_act import ConditionedNormActBlock
from mypt.building_blocks.auxiliary.normalization.utils import get_normalization
from mypt.building_blocks.auxiliary.activations.activations import get_activation


class TwoDimFiLMBlock(ConditionedNormActBlock):
    """
    A FiLM block is a block that applies a FiLM transformation to the input.
    """

    def __init__(self, 
                normalization: Union[str, Callable],
                normalization_params: dict,
                activation: Union[str, Callable],
                activation_params: dict,
                out_channels: int,
                cond_channels: int,
                hidden_units: int = 256,
                film_activation: Union[str, Callable] = "relu",
                film_activation_params: dict = {}
                ):
        
        super().__init__(normalization, normalization_params, activation, activation_params, ['_film_layer', '_normalization', '_activation'])

        self._film_layer = nn.Sequential(
            nn.Linear(cond_channels, hidden_units),
            get_activation(film_activation, film_activation_params),
            nn.Linear(hidden_units, 2 * out_channels)
        )
        self._normalization = get_normalization(normalization, normalization_params)
        self._activation = get_activation(activation, activation_params)    



    def forward(self, 
                x: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        
        film_params = self._film_layer(condition) # film_params of the shape (batch_size, 2 * out_channels, height, width)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # this function call returns two tensors of the shape (batch_size, out_channels, height, width)
        
        # since gamma and beta are 2 dimensional, they should be reshaped for broadcasting
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)

        out = self._normalization(x)
        out = out * gamma + beta
        return self._activation(out)
    

class ThreeDimFiLMBlock(ConditionedNormActBlock):
    """
    A FiLM block is a block that applies a FiLM transformation to the input.
    """

    def __init__(self, 
                normalization: Union[str, Callable],
                normalization_params: dict,
                activation: Union[str, Callable],
                activation_params: dict,
                out_channels: int,
                cond_channels: int,
                hidden_channels: int = 256,
                film_activation: Union[str, Callable] = "relu",
                film_activation_params: dict = {}
                ):
        
        super().__init__(normalization, normalization_params, activation, activation_params, ['_film_layer', '_normalization', '_activation'])

        self._film_layer = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            get_activation(film_activation, film_activation_params),
            nn.Conv2d(hidden_channels, 2 * out_channels, kernel_size=3, padding=1, stride=1)
        )
        self._normalization = get_normalization(normalization, normalization_params)
        self._activation = get_activation(activation, activation_params)


    def forward(self, 
                x: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        
        film_params = self._film_layer(condition) # film_params of the shape (batch_size, 2 * out_channels, height, width)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # this function call returns two tensors of the shape (batch_size, out_channels, height, width)
        
        out = self._normalization(x)
        out = out * gamma + beta
        return self._activation(out)

