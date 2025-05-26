import torch

from torch import nn
from typing import List, Optional, Union, Callable

from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet_layers import AbstractDownLayer, AbstractUpLayer
from mypt.building_blocks.conv_blocks.conditioned.three_dim.resnet_con3d import (
    CondThreeDimWResBlock, 
    UpCondThreeDimWResBlock,
    DownCondThreeDimWResBlock
)


class UnetDownLayer3D(AbstractDownLayer):
    """
    Downsampling layer for UNet architecture using DownLayer components.
    """
    def __init__(self, 
        num_resnet_blocks: int,
        in_channels: int,
        cond_channels: int,
        out_channels: List[int],
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
        downsample_type: str = 'con'):

        super().__init__(
            in_channels,
            cond_channels,
            out_channels,
            num_resnet_blocks,
        )

        resnet_parameters = {
            'norm1': norm1,
            'norm1_params': norm1_params,
            'norm2': norm2,
            'norm2_params': norm2_params,
            
            'activation': activation,
            'activation_params': activation_params,
            'film_activation': film_activation, 
            'film_activation_params': film_activation_params,
            'force_residual': force_residual,
            'dropout_rate': dropout_rate,
            'inner_dim': inner_dim,
            'downsample_type': downsample_type,
        }

        resnet_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondThreeDimWResBlock(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                cond_channels=cond_channels,
                **resnet_parameters
            )

        resnet_blocks[-1] = DownCondThreeDimWResBlock(
            in_channels=out_channels[-1],
            out_channels=out_channels[-1],
            cond_channels=cond_channels,
            **resnet_parameters
        )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # just use the super class forward method 
        return super().forward(x, condition)
    

class UnetUpLayer3D(AbstractUpLayer):
    """
    Upsampling layer for UNet architecture using UpLayer components.
    """
    def __init__(self, 
        num_resnet_blocks: int,
        in_channels: int,
        cond_channels: int,
        out_channels: List[int],
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
        upsample_type: str = 'con'):

        super().__init__(
            in_channels,
            cond_channels,
            out_channels,
            num_resnet_blocks,
        )

        resnet_parameters = {
            'norm1': norm1,
            'norm1_params': norm1_params,
            'norm2': norm2,
            'norm2_params': norm2_params,
            'activation': activation,
            'activation_params': activation_params,
            'film_activation': film_activation,
            'film_activation_params': film_activation_params,
            'force_residual': force_residual,
            'dropout_rate': dropout_rate,
            'inner_dim': inner_dim,
            'upsample_type': upsample_type,
        }

        resnet_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondThreeDimWResBlock(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                cond_channels=cond_channels,
                **resnet_parameters
            )

        resnet_blocks[-1] = UpCondThreeDimWResBlock(
            in_channels=out_channels[-1],
            out_channels=out_channels[-1],
            cond_channels=cond_channels,
            **resnet_parameters
        )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # just use the super class forward method 
        return super().forward(x, condition)

    