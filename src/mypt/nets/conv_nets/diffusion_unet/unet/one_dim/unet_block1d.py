import torch
import torch.nn as nn

from typing import Iterator, List, Union, Optional, Callable, Tuple

from mypt.building_blocks.mixins.general import SequentialModuleListMixin
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet_layer1d import UnetDownLayer1D, UnetUpLayer1D
from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import CondOneDimWResBlock
from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet_blocks import AbstractUnetDownBlock, AbstractUnetUpBlock

class UnetDownBlock1D(AbstractUnetDownBlock):
    """
    Downsampling block for UNet architecture using DownLayer components.
    """
    def __init__(self, 
                 num_down_layers: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_dimension: int,
                 out_channels: Union[int, List[int]],
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
                 *args, **kwargs):
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

        # Create the down layers
        down_layers = [None for _ in range(num_down_layers)]

        for i in range(num_down_layers):
            down_layers[i] = UnetDownLayer1D(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                downsample_type=downsample_types[i],
                **self._layer_params
            )
        # set the self.down_layers field
        self.down_layers = nn.ModuleList(down_layers)



    # override the forward method
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return super().forward(x, condition)
        # skip_connections = [None for _ in self.down_layers]

        # for i, layer in enumerate(self.down_layers):
        #     x, skip = layer(x, condition)
        #     skip_connections[i] = skip

        # return x, skip_connections



class UnetUpBlock1D(AbstractUnetUpBlock):
    """
    Upsampling block for UNet architecture using UpLayer components.
    """
    def __init__(self, 
                 num_up_blocks: int,
                 num_resnet_blocks: int,
                 in_channels: int,
                 cond_channels: int,
                 out_channels: Union[int, List[int]],
                 upsample_types: Union[str, List[str]] = "transpose_conv", 
                 inner_dim: int = 256,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the up layers
        up_layers = [None for _ in range(num_up_blocks)]

        for i in range(num_up_blocks):
            up_layers[i] = UnetUpLayer1D(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                upsample_type=upsample_types[i],
                **self._layer_params
            )

        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, x: torch.Tensor, skip_outputs: List[torch.Tensor], condition: torch.Tensor) -> torch.Tensor:
        return super().forward(x, skip_outputs, condition)


