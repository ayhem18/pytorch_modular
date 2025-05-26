import torch
import torch.nn as nn

from typing import Iterator, List, Union, Optional, Callable, Tuple


from mypt.building_blocks.mixins.general import SequentialModuleListMixin
from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import (
    CondOneDimWResBlock,
    DownCondOneDimWResBlock,
    UpCondOneDimWResBlock
)


class DownLayer(SequentialModuleListMixin):
    """
    A single downsampling layer for UNet architecture.
    
    This consists of:
    1. A series of CondOneDimWResBlock instances
    2. An optional DownCondOneDimWResBlock if downsampling is enabled
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_channels: Number of conditioning channels
        num_resnet_blocks: Number of residual blocks to use
        downsample: Whether to include downsampling at the end
        downsample_type: Type of downsampling to use ("conv", "avg_pool", "max_pool")
        inner_dim: Inner dimension for FiLM layer
        dropout_rate: Dropout rate
        norm1: Normalization for the first residual block
        norm1_params: Parameters for the first normalization
        norm2: Normalization for the second residual block
        norm2_params: Parameters for the second normalization
        activation: Activation function
        activation_params: Parameters for activation function
        film_activation: Activation function for FiLM
        film_activation_params: Parameters for FiLM activation
        force_residual: Whether to force residual connections
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        num_resnet_blocks: int,
        downsample_type: str = "conv",
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
    ):
        super().__init__("_resnet_blocks")

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Store common parameters for the resnet blocks
        resnet_params = {
            "inner_dim": inner_dim,
            "dropout_rate": dropout_rate,
            "norm1": norm1,
            "norm1_params": norm1_params,
            "norm2": norm2,
            "norm2_params": norm2_params,
            "activation": activation,
            "activation_params": activation_params,
            "film_activation": film_activation,
            "film_activation_params": film_activation_params,
            "force_residual": force_residual,
        }
        
        # Create the resnet blocks
        resnet_blocks = [_ for _ in range(num_resnet_blocks)]
        
        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondOneDimWResBlock(
                # the first CondOneDimWResBlock is expected to have the same number of channels as the input, the rest are expected to have the same number of channels as the output 
                in_channels=in_channels if i == 0 else out_channels, 
                out_channels=out_channels,
                cond_channels=cond_channels,
                **resnet_params
            )


        resnet_blocks[num_resnet_blocks - 1] = DownCondOneDimWResBlock(
                                in_channels=out_channels,
                                out_channels=out_channels,
                                cond_channels=cond_channels,
                                downsample_type=downsample_type,
                                **resnet_params
                                )
    
        self.resnet_blocks = nn.ModuleList(resnet_blocks)



    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the down layer.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor
            
        Returns:
            List of tensors from each resnet block
        """
        return super().sequential_module_list_forward(x, condition)

    def to(self, *args, **kwargs) -> 'DownLayer': 
        super().module_list_to(*args, **kwargs)
        return self 
    
    def train(self, mode: bool = True) -> 'DownLayer':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'DownLayer':
        super().module_list_eval()
        return self
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()

    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)





class UpLayer(SequentialModuleListMixin):
    """
    A single upsampling layer for UNet architecture.
    
    This consists of:
    1. An optional UpCondWResnetBlock if upsampling is enabled
    2. A series of ConditionalWResBlock instances
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_channels: Number of conditioning channels
        num_resnet_blocks: Number of residual blocks to use
        film_dimension: Dimension for FiLM conditioning (2 or 3)
        upsample: Whether to include upsampling at the beginning
        upsample_type: Type of upsampling to use ("transpose_conv", "conv", "interpolate")
        inner_dim: Inner dimension for FiLM layer
        dropout_rate: Dropout rate
        norm1: Normalization for the first residual block
        norm1_params: Parameters for the first normalization
        norm2: Normalization for the second residual block
        norm2_params: Parameters for the second normalization
        activation: Activation function
        activation_params: Parameters for activation function
        film_activation: Activation function for FiLM
        film_activation_params: Parameters for FiLM activation
        force_residual: Whether to force residual connections
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        num_resnet_blocks: int,
        upsample_type: str = "transpose_conv",
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
    ):  
        super().__init__("_resnet_blocks")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Store common parameters for the resnet blocks
        resnet_params = {
            "inner_dim": inner_dim,
            "dropout_rate": dropout_rate,
            "norm1": norm1,
            "norm1_params": norm1_params,
            "norm2": norm2,
            "norm2_params": norm2_params,
            "activation": activation,
            "activation_params": activation_params,
            "film_activation": film_activation,
            "film_activation_params": film_activation_params,
            "force_residual": force_residual,
        }
        
        resnet_blocks = [_ for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks - 1):
            resnet_blocks[i] = CondOneDimWResBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                cond_channels=cond_channels,
                **resnet_params
            )


        resnet_blocks[-1] = UpCondOneDimWResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            upsample_type=upsample_type,
            **resnet_params
        )

        self.resnet_blocks = nn.ModuleList(resnet_blocks)

  
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up layer.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor
            condition: Conditioning tensor
            
        Returns:
            Output tensor
        """
        return super().sequential_module_list_forward(x, condition)
    
    def to(self, *args, **kwargs) -> 'UpLayer': 
        super().module_list_to(*args, **kwargs)
        return self 
    
    def train(self, mode: bool = True) -> 'UpLayer':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'UpLayer':
        super().module_list_eval()
        return self 
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse) 
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)
