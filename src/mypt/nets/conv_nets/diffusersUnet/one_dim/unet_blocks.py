import torch
import torch.nn as nn
from typing import Iterator, List, Union, Optional, Callable, Tuple

from mypt.nets.conv_nets.diffusersUnet.one_dim.unet_layers import DownLayer, UpLayer
from mypt.building_blocks.mixins.general import ModuleListMixin, SequentialModuleListMixin
from mypt.building_blocks.conv_blocks.conditioned.one_dim.resnet_con1d import CondOneDimWResBlock


class UnetDownBlock(ModuleListMixin):
    """
    Downsampling block for UNet architecture using DownLayer components.
    
    This block consists of multiple downsampling stages, each implemented as a DownLayer.
    
    Args:
        num_down_blocks: Number of downsampling stages
        num_resnet_blocks: Number of residual blocks per stage
        in_channels: Input channels 
        cond_channels: Conditioning channels
        out_channels: List of output channels for each stage
        downsample_types: Downsampling method(s) to use
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
        num_down_blocks: int,
        num_resnet_blocks: int,
        in_channels: int,
        cond_channels: int,
        out_channels: List[int],
        film_dimension: int,
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
        super().__init__("_down_layers")
        
        # Validate inputs
        if len(out_channels) != num_down_blocks:
            raise ValueError(f"Expected out_channels to have {num_down_blocks} elements, but got {len(out_channels)}")
        
        # Convert downsample_types to list if it's a string
        if isinstance(downsample_types, str):
            downsample_types = [downsample_types] * num_down_blocks
        elif len(downsample_types) != num_down_blocks:
            raise ValueError(f"Expected downsample_types to have {num_down_blocks} elements, but got {len(downsample_types)}")
        
        # Store parameters
        self.num_down_blocks = num_down_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Common parameters for all layers
        self.layer_params = {
            "num_resnet_blocks": num_resnet_blocks,
            "cond_channels": cond_channels,
            "film_dimension": film_dimension,
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
        
        # Create the down layers
        down_layers = [None for _ in range(num_down_blocks)]

        for i in range(num_down_blocks):
            down_layers[i] = DownLayer(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                downsample_type=downsample_types[i],
                **self.layer_params
            )

        self.down_layers = nn.ModuleList(down_layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the down block.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor
            
        Returns:
            Tuple containing:
            - Final output tensor 
            - List of skip connection tensors for each layer
        """
        skip_connections = [None for _ in self.down_layers]
        
        for i, layer in enumerate(self.down_layers):
            x, skip = layer(x, condition)
            skip_connections[i] = skip
        
        return x, skip_connections
    
    def to(self, *args, **kwargs) -> 'UnetDownBlock':
        super().module_list_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'UnetDownBlock':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'UnetDownBlock':
        super().module_list_eval()
        return self
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()
    

    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)
    
    
    



class UnetUpBlock(ModuleListMixin):
    """
    Upsampling block for UNet architecture using UpLayer components.
    
    This block consists of multiple upsampling stages, each implemented as an UpLayer.
    
    Args:
        num_up_blocks: Number of upsampling stages
        num_resnet_blocks: Number of residual blocks per stage
        in_channels: Input channels
        cond_channels: Conditioning channels
        out_channels: List of output channels for each stage
        film_dimension: Dimension for FiLM conditioning
        upsample_types: Upsampling method(s) to use
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
        num_up_blocks: int,
        num_resnet_blocks: int,
        in_channels: int,
        cond_channels: int,
        out_channels: List[int],
        film_dimension: int,
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
        super().__init__()
        
        # Validate inputs
        if len(out_channels) != num_up_blocks:
            raise ValueError(f"Expected out_channels to have {num_up_blocks} elements, but got {len(out_channels)}")
        
        # Convert upsample_types to list if it's a string
        if isinstance(upsample_types, str):
            upsample_types = [upsample_types] * num_up_blocks
        elif len(upsample_types) != num_up_blocks:
            raise ValueError(f"Expected upsample_types to have {num_up_blocks} elements, but got {len(upsample_types)}")
        
        # Store parameters
        self.num_up_blocks = num_up_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Common parameters for all layers
        self.layer_params = {
            "num_resnet_blocks": num_resnet_blocks,
            "cond_channels": cond_channels,
            "film_dimension": film_dimension,
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
        
        up_layers = [None for _ in range(num_up_blocks)]

        for i in range(num_up_blocks):            
            up_layers[i] = UpLayer(
                in_channels=in_channels if i == 0 else out_channels[i-1],
                out_channels=out_channels[i],
                upsample_type=upsample_types[i],
                **self.layer_params
            )
    
        self.up_layers = nn.ModuleList(up_layers)
    
    def forward(self, x: torch.Tensor, skip_outputs: List[torch.Tensor], condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up block.
        
        Args:
            x: Input tensor [B, C, H, W]
            skip_connections: List of skip connection tensors from the down block
            condition: Conditioning tensor
            
        Returns:
            Output tensor
        """
        if len(skip_outputs) != len(self.up_layers):
            raise ValueError(f"Expected skip_outputs to have {len(self.up_layers)} elements, but got {len(skip_outputs)}")
        
        # Reverse skip connections to match the order of up layers
        skip_outputs = skip_outputs[::-1]

        for i, layer in enumerate(self.up_layers):
            if skip_outputs[i].shape != x.shape:
                raise ValueError(f"Expected skip_outputs[{i}] to have shape {x.shape}, but got {skip_outputs[i].shape}")

            layer_input = x + skip_outputs[i]
            x = layer(layer_input, condition)
        
        return x


class UNetMidBlock(SequentialModuleListMixin):
    """
    Middle block for UNet architecture.
    
    This block consists of a sequence of ConditionalWResBlock instances.
    
    Args:
        num_resnet_blocks: Number of residual blocks
        in_channels: Input channels
        film_dimension: Dimension for FiLM conditioning
        out_channels: Output channels (default: same as input)
        cond_channels: Conditioning channels
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
        num_resnet_blocks: int,
        in_channels: int,
        film_dimension: int,
        out_channels: Optional[int] = None,
        cond_channels: int = 512,
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
        super().__init__()
        
        out_channels = out_channels or in_channels
        
        # Store parameters
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        
        # Create the mid blocks directly as a sequence of ConditionalWResBlock
        mid_blocks = [None for _ in range(num_resnet_blocks)]

        for i in range(num_resnet_blocks):
            mid_blocks[i] = CondOneDimWResBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                cond_channels=cond_channels,
                inner_dim=inner_dim,
                dropout_rate=dropout_rate,
                norm1=norm1,
                norm1_params=norm1_params,
                norm2=norm2,
                norm2_params=norm2_params,
                activation=activation,
                activation_params=activation_params,
                film_activation=film_activation,
                film_activation_params=film_activation_params,
                force_residual=force_residual,
            )

        self.mid_blocks = nn.ModuleList(mid_blocks)
    

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return super().sequential_module_list_forward(x, condition)
    
    def to(self, *args, **kwargs) -> 'UNetMidBlock':
        super().module_list_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'UNetMidBlock':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'UNetMidBlock':
        super().module_list_eval()
        return self
    
    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return super().module_list_parameters(recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return super().module_list_named_parameters(prefix, recurse)
