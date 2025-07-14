import torch
import torch.nn as nn

from typing import List, Union, Optional, Callable, Tuple

from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import CondOneDimWResBlock
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet_layers1d import UnetDownLayer1D, UnetUpLayer1D
from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet_blocks import AbstractUnetDownBlock, AbstractUnetUpBlock, AbstractUnetMidBlock


class UnetDownBlock1D(AbstractUnetDownBlock):
    """
    Downsampling block for UNet architecture using DownLayer components.
    """
    def __init__(self, 
                 num_down_layers: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_dimension: int,
                 out_channels: Union[List[int], List[List[int]]],
                 downsample_types: Union[str, List[str]] = "conv",  
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
                 *args, **kwargs
                 ):
        super().__init__(
            num_down_layers,
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            downsample_types,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,
            *args, **kwargs
        )

        down_layers = [
            UnetDownLayer1D(
                            in_channels=self.in_channels if i == 0 else self.out_channels[i-1][-1],
                            out_channels=self.out_channels[i],
                            downsample_type=self.downsample_types[i],
                            **self._layer_params
                        )   
            for i in range(num_down_layers)
        ]

        self.down_layers = nn.ModuleList(down_layers)


    # override the forward method
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return super().forward(x, condition)


class UnetUpBlock1D(AbstractUnetUpBlock):   
    """
    Upsampling block for UNet architecture using UpLayer components.
    """
    def __init__(self, 
                 num_up_layers: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_dimension: int,
                 out_channels: Union[List[int], List[List[int]]],
                 skip_connections_channels: List[int],
                 reverse_skip_connections: bool = True,
                 upsample_types: Union[str, List[str]] = "transpose_conv",  
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
                 *args, **kwargs):

        super().__init__(
            num_up_layers,
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            skip_connections_channels,
            reverse_skip_connections,
            upsample_types,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,   
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,      
            *args, **kwargs
        )


        up_layers = [
            UnetUpLayer1D(
                in_channels=self.up_sample_in_channels[i],
                out_channels=self.out_channels[i],
                upsample_type=self.upsample_types[i],
                **self._layer_params
            )
            for i in range(num_up_layers)
        ]

        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, x: torch.Tensor, skip_outputs: List[torch.Tensor], condition: torch.Tensor, reverse_skip_connections: bool = True) -> torch.Tensor:
        return super().forward(x, skip_outputs, condition, reverse_skip_connections)


class UNetMidBlock1D(AbstractUnetMidBlock):
    """
    Middle block for UNet architecture using CondResnet components.
    """
    def __init__(self, 
                num_resnet_blocks: int,
                in_channels: int,
                cond_dimension: int,
                out_channels: Optional[int] = None,
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
                *args, **kwargs):
        super().__init__(
            num_resnet_blocks,
            in_channels,
            cond_dimension,
            out_channels,
            inner_dim,
            dropout_rate,
            norm1,
            norm1_params,
            norm2,
            norm2_params,
            activation,
            activation_params,
            film_activation,
            film_activation_params,
            force_residual,
            *args, **kwargs
        )

        mid_blocks = [
            CondOneDimWResBlock(
                in_channels=self.in_channels if i == 0 else self.out_channels,
                out_channels=self.out_channels,
                cond_dimension=self.cond_dimension,
                stride=1, # the blocks should not change the spatial dimensions
                **self._block_params
            )
            for i in range(num_resnet_blocks)
        ]

        self._mid_blocks = nn.ModuleList(mid_blocks)

