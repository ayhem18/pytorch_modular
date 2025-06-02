import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Callable

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin


class AbstractUNetCond(NonSequentialModuleMixin, nn.Module, ABC):
    """
    Abstract UNet architecture implementation using the builder design pattern.
    
    This UNet consists of three main components:
    1. Down Block: Series of downsampling layers
    2. Middle Block: Processing at the bottleneck
    3. Up Block: Series of upsampling layers
    
    Each component can be configured separately using the build_* methods.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        cond_dimension: Dimension of the conditioning input
    """
    def __init__(self, in_channels: int, out_channels: int, cond_dimension: int, *args, **kwargs):
        torch.nn.Module.__init__(self, *args, **kwargs)
        NonSequentialModuleMixin.__init__(self, ["_down_block", "_middle_block", "_up_block"])
        
        self.input_channels: int = in_channels # the number of channels of the Unet input 
        self.final_out_channels: int = out_channels # the number of channels of the Unet output 
        self.cond_dimension = cond_dimension # the dimension of the conditioning input

        # the number of downsampling layers 
        self.num_down_layers: int = None # the number of downsampling layers 

        # the out channels of the unetDownBlock 
        self.down_block_out_channels: Union[List[int], List[List[int]]] = None
        # the number of resnet blocks in the unetDownBlock 
        self.down_block_num_resnet_blocks: int = None
        
        # the number of middle resnet blocks in the middle block 
        self.middle_block_num_resnet_blocks: int = None 

        # the number of resnet blocks in the unetUpBlock 
        self.up_block_num_resnet_blocks: int = None

        # Initialize components as None
        self._down_block = None
        self._middle_block = None
        self._up_block = None

        self._is_built: bool = False
    
    @abstractmethod
    def build_down_block(
        self,
        num_down_layers: int,
        num_resnet_blocks: int,
        out_channels: List[int],
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
    ):
        """
        Configure and build the downsampling block.
        
        Args:
            num_down_layers: Number of downsampling layers
            num_resnet_blocks: Number of residual blocks per stage
            out_channels: List of output channels for each stage
            downsample_types: Type of downsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        pass
    
    @abstractmethod
    def build_middle_block(
        self,
        num_resnet_blocks: int,
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
        """
        Configure and build the middle block.
        
        Args:
            num_resnet_blocks: Number of residual blocks
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        pass
    
    @abstractmethod
    def build_up_block(
        self,
        num_resnet_blocks: int,
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
    ):
        """
        Configure and build the upsampling block.
        
        Args:
            num_resnet_blocks: Number of residual blocks per stage
            upsample_types: Type of upsampling to use
            inner_dim: Inner dimension for FiLM layer
            dropout_rate: Dropout rate
            norm1, norm2: Normalization layers
            norm1_params, norm2_params: Parameters for normalization layers
            activation: Activation function
            activation_params: Parameters for activation function
            film_activation: Activation function for FiLM
            film_activation_params: Parameters for FiLM activation
            force_residual: Whether to force residual connections
        """
        pass
    
    def _verify_input(self, x: torch.Tensor) -> None:
        """
        Verify that the input tensor has dimensions compatible with the UNet architecture.
        
        The height and width must be divisible by 2^num_down_layers.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Raises:
            ValueError: If height or width is not divisible by 2^num_down_layers
        """
        num_2_expo_height = 0
        num_2_expo_width = 0

        h, w = x.shape[2:]

        while h % 2 == 0:
            num_2_expo_height += 1
            h = h // 2

        while w % 2 == 0:
            num_2_expo_width += 1
            w = w // 2
        
        if min(num_2_expo_height, num_2_expo_width) < self.num_down_layers:
            raise ValueError(f"The UNet architecture requires that the input height and width are divisible by 2^(number of downsampling layers). Otherwise, the skip connections would get fairly complex...")

    @abstractmethod
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor
            condition: Conditioning tensor
            
        Returns:
            Output tensor
        """
        pass

    def __call__(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.forward(x, condition) 