import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Optional, Union, Callable, Iterator, Tuple

from mypt.building_blocks.auxiliary.film_block import TwoDimFiLMBlock
from mypt.building_blocks.mixins.residual_mixins import NonSequentialModuleMixin


class ConditionedWideResnetBlock(NonSequentialModuleMixin):
    """
    A conditioned version of the WideResnet block that incorporates feature-wise
    conditioning through FiLM (Feature-wise Linear Modulation).
    
    This block follows the same architectural principles as WideResnetBlock but 
    allows for external conditioning information to modulate the features.
    
    The structure includes:
    1. FiLM-conditioned normalization and activation
    2. Convolution
    3. Dropout
    4. FiLM-conditioned normalization and activation
    5. Convolution
    6. Residual connection (either identity or 3x3 conv)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        hidden_units: int = 256,
        stride: int = 1, 
        dropout_rate: float = 0.0,

        # normalization parameters  
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
       
        # activation parameters
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        
        # FiLM activation parameters
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},

        # whether to use a convolutional layer as the shortcut connection
        force_residual: bool = False
    ):
        # Validate stride
        if stride not in [1, 2]:
            raise ValueError(f"Stride must be 1 or 2, got {stride}")
       # Initialize the mixin
        super().__init__(
            inner_components_fields=[
                "_components",  
                "_shortcut"
            ]
        )
        
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_channels = cond_channels
        self._stride = stride
        self._dropout_rate = dropout_rate
        self._hidden_units = hidden_units
        self._activation = activation
        self._activation_params = activation_params
        self._film_activation = film_activation
        self._film_activation_params = film_activation_params
        
        # Prepare normalization params
        norm1_params = norm1_params or {}
        norm1_params["in_channels"] = in_channels
        
        norm2_params = norm2_params or {}
        norm2_params["in_channels"] = out_channels
        
        # Create main stream components
        # We'll use ModuleDict instead of OrderedDict to store the layers
        # because the order of forward pass will be handled explicitly
        self._components = nn.ModuleDict({
            # First FiLM block
            'film1': TwoDimFiLMBlock(
                normalization=norm1,
                normalization_params=norm1_params,
                activation=activation,
                activation_params=activation_params,
                out_channels=in_channels,
                cond_channels=cond_channels,
                hidden_units=hidden_units,
                film_activation=film_activation,
                film_activation_params=film_activation_params
            ),
            # First convolution
            'conv1': nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
            ),
            # Dropout
            'dropout': nn.Dropout(dropout_rate),
            # Second FiLM block
            'film2': TwoDimFiLMBlock(
                normalization=norm2,
                normalization_params=norm2_params,
                activation=activation,
                activation_params=activation_params,
                out_channels=out_channels,
                cond_channels=cond_channels,
                hidden_units=hidden_units,
                film_activation=film_activation,
                film_activation_params=film_activation_params
            ),
            # Second convolution
            'conv2': nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1,  # Always 1 for the second conv
                padding=1, 
            )
        })
        
        # Create shortcut connection if needed
        if force_residual:
            self._shortcut = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1,
            )
    
    def _forward_main_stream(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the main stream with conditioning.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            condition: Conditioning tensor [B, cond_dim]
            
        Returns:
            Output tensor after passing through the main stream
        """
        # First FiLM -> Conv -> Dropout sequence
        out = self._components['film1'](x, condition)
        out = self._components['conv1'](out)
        out = self._components['dropout'](out)
        
        # Second FiLM -> Conv sequence
        out = self._components['film2'](out, condition)
        out = self._components['conv2'](out)
        
        return out
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the conditioned residual block.
        
        Args:
            x: Input tensor [B, C, H, W]
            condition: Conditioning tensor [B, cond_dim]
            debug: If True, returns intermediate tensors
            
        Returns:
            Output tensor or (main_output, residual_output, final_output) if debug=True
        """
        # Process the main stream
        main_stream_output = self._forward_main_stream(x, condition)
        
        # If no residual connection, add input directly
        if self._shortcut is None:
            if main_stream_output.shape != x.shape:
                raise ValueError(f"Main stream output shape {main_stream_output.shape} doesn't match input shape {x.shape}")
            
            if debug:
                return main_stream_output, x, main_stream_output + x
            
            return main_stream_output + x
        
        # Process the residual stream
        residual_stream_output = self._shortcut(x)
        
        if residual_stream_output.shape != main_stream_output.shape:
            raise ValueError(f"Residual output shape {residual_stream_output.shape} doesn't match main stream shape {main_stream_output.shape}")
        
        if debug:
            return main_stream_output, residual_stream_output, main_stream_output + residual_stream_output
        
        return main_stream_output + residual_stream_output
    
    def __call__(self, x: torch.Tensor, condition: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Support for optional debug parameter in __call__"""
        return self.forward(x, condition, debug=debug)
    
    # Override standard module methods
    def children(self) -> Iterator[nn.Module]:
        """Returns an iterator over immediate children modules."""
        yield self._components
        if hasattr(self, "_shortcut") and self._shortcut is not None:
            yield self._shortcut
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module parameters."""
        for name, module in self.named_children():
            for param in module.parameters(recurse):
                yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Returns an iterator over module parameters, yielding both name and parameter."""
        components_prefix = prefix + '.components.' if prefix else 'components.'
        for name, param in self._components.named_parameters(components_prefix, recurse):
            yield name, param
            
        if hasattr(self, "_shortcut") and self._shortcut is not None:
            shortcut_prefix = prefix + '.shortcut.' if prefix else 'shortcut.'
            for name, param in self._shortcut.named_parameters(shortcut_prefix, recurse):
                yield name, param
    
    def to(self, *args, **kwargs) -> 'ConditionedWideResnetBlock':
        """Moves and/or casts the parameters and buffers."""
        self._components.to(*args, **kwargs)
        if hasattr(self, "_shortcut") and self._shortcut is not None:
            self._shortcut.to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'ConditionedWideResnetBlock':
        """Sets the module in training mode."""
        self.training = mode
        self._components.train(mode)
        if hasattr(self, "_shortcut") and self._shortcut is not None:
            self._shortcut.train(mode)
        return self
    
    def eval(self) -> 'ConditionedWideResnetBlock':
        """Sets the module in evaluation mode."""
        return self.train(False)
    
    # Properties
    @property
    def in_channels(self) -> int:
        """Returns the number of input channels."""
        return self._in_channels
    
    @property
    def out_channels(self) -> int:
        """Returns the number of output channels."""
        return self._out_channels
    
    @property
    def cond_channels(self) -> int:
        """Returns the number of conditioning channels."""
        return self._cond_channels
    
    @property
    def stride(self) -> int:
        """Returns the stride used in the block."""
        return self._stride
    
    @property
    def dropout_rate(self) -> float:
        """Returns the dropout rate used in the block."""
        return self._dropout_rate
