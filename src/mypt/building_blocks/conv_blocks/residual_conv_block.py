import torch
from torch import nn
from typing import Iterator, List, Tuple, Union, Optional

from mypt.building_blocks.mixins.residual_mixins import GeneralResidualMixin
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin, CloneableModuleMixin
from mypt.building_blocks.conv_blocks.conv_block import BasicConvBlock
from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser


class ResidualConvBlock(GeneralResidualMixin, WrapperLikeModuleMixin):
    """
    A residual convolutional block that adds a skip connection around a BasicConvBlock.
    
    If input and output dimensions differ, an adaptive convolution layer is created.
    If dimensions match and force_residual=True, a 1x1 convolution is used as the skip connection.
    Otherwise, a direct identity connection is used.
    """

    def __init__(self, 
                 num_conv_layers: int, 
                 channels: Union[List[int], Tuple[int]],
                 kernel_sizes: Union[List[int], int],
                 strides: Optional[Union[List[int], int]] = 1,
                 paddings: Optional[Union[List[int], List[str], str, int]] = 'same',
                 use_bn: bool = True,
                 activation_after_each_layer: bool = True,
                 activation: Optional[nn.Module] = None,
                 activation_params: Optional[dict] = None,
                 final_bn_layer: bool = False,
                 force_residual: bool = False,
                 *args, **kwargs):
        """
        Initialize a ResidualConvBlock.
        
        Args:
            num_conv_layers: Number of convolutional layers in the block
            channels: List of channel dimensions for each layer
            kernel_sizes: Kernel size(s) for the convolutions
            strides: Stride(s) for the convolutions
            paddings: Padding value(s) for the convolutions
            use_bn: Whether to use batch normalization
            activation_after_each_layer: Whether to apply activation after each conv layer
            activation: Activation function to use
            activation_params: Parameters for the activation function
            final_bn_layer: Whether to add a batch norm layer at the end
            force_residual: Whether to force a residual connection even when dimensions match
        """
        WrapperLikeModuleMixin.__init__(self, '_block', *args, **kwargs)
        
        # Store constructor parameters
        self._num_conv_layers = num_conv_layers
        self._channels = channels
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        self._use_bn = use_bn
        self._activation_after_each_layer = activation_after_each_layer
        self._activation = activation
        self._activation_params = activation_params
        self._final_bn_layer = final_bn_layer
        self._force_residual = force_residual
        
        # Create the main convolutional block
        self._block = BasicConvBlock(
            num_conv_layers=num_conv_layers,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_bn=use_bn,
            activation_after_each_layer=activation_after_each_layer,
            activation=activation,
            activation_params=activation_params,
            final_bn_layer=final_bn_layer
        )
        
        # Determine if we need a residual connection
        self._adaptive_layer = None
        
        # Use dimension analyzer to determine output shape
        dim_analyzer = DimensionsAnalyser(method='static') # use static method: faster and memory efficient
        
        # Create a sample input tensor with reasonable dimensions
        # The actual values don't matter, just the shape
        sample_height, sample_width = 1024, 1024
        sample_input = torch.zeros(1, channels[0], sample_height, sample_width)
        
        # Get the output shape
        output_shape = dim_analyzer.analyse_dimensions(sample_input.shape, self._block)
        
        if output_shape[1:] != (channels[-1], sample_height, sample_width) or force_residual:
            # the adaptive layer should produce the output shape from the input shape
            # the input shape is (1, channels[0], sample_height, sample_width)
            # the output shape is (1, channels[-1], sample_height, sample_width)
            # the kernel size should be (output_shape[2] - sample_height + 1, output_shape[3] - sample_width + 1)   
            # if the shape are exactly the same (and force_residual is True), then the formula is still correct

            self._adaptive_layer = nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[-1],
                kernel_size=(output_shape[2] - sample_height + 1, output_shape[3] - sample_width + 1), 
                stride=1,
                padding=0
            )
            
            # Set up GeneralResidualMixin with both streams
            GeneralResidualMixin.__init__(
                self,
                main_stream_field_name='_block',
                residual_stream_field_name='_adaptive_layer'
            )
            
            return 
        
        # No adaptive layer needed, use identity connection
        GeneralResidualMixin.__init__(
            self,
            main_stream_field_name='_block',
            residual_stream_field_name=None
        )

    def forward(self, x: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            debug: If True, returns main stream output, residual stream output, and final output
            
        Returns:
            Output tensor or (main_output, residual_output, final_output) if debug=True
        """
        return self.residual_forward(x, debug=debug)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    # Override methods with residual implementations
    def children(self) -> Iterator[nn.Module]:
        return self.residual_children()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.residual_named_children()
    
    def modules(self) -> Iterator[nn.Module]:
        return self.residual_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.residual_parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        return self.residual_named_parameters(prefix=prefix, recurse=recurse)
    
    def to(self, *args, **kwargs) -> 'ResidualConvBlock':
        return self.residual_to(*args, **kwargs)
    
    def train(self, mode: bool = True) -> 'ResidualConvBlock':
        return self.residual_train(mode)
    
    def eval(self) -> 'ResidualConvBlock':
        return self.residual_eval()


    @property
    def adaptive_layer(self) -> Optional[nn.Conv2d]:
        return self._adaptive_layer
    
    @property
    def block(self) -> BasicConvBlock:
        return self._block
    

