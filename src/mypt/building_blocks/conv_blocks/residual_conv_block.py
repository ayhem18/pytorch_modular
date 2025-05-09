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
                 input_shape: Optional[Tuple[int, int, int]] = None,  # (C, H, W)
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
            input_shape: Optional input shape (C, H, W) needed for non-trivial kernel sizes
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
        self._input_shape = input_shape
        
        # Validate kernel sizes
        # If kernel sizes include values > 1, we need input_shape to properly create the adaptive layer
        if isinstance(strides, int):
            all_strides_are_one = (strides == 1)
        else:
            all_strides_are_one = all(s == 1 for s in strides)
            
        if not all_strides_are_one and input_shape is None:
            raise ValueError("When using strides > 1, input_shape must be provided. "
                             "Strided convolutions are non-linear operations that cannot "
                             "always be represented by an equivalent single convolution layer  in the residual stream !!!.")
        
        # the final step is to make sure that the input shape[0] is the same as the channels[0]
        if input_shape is not None and input_shape[0] != channels[0]:
            raise ValueError(f"The input shape[0] ({input_shape[0]}) must be the same as the channels[0] ({channels[0]}).")

        # one final check: if the stride is set 1, then the kernel size must be larger than the padding
        # to guarantee that the output shape is smaller than the input shape

        # normalize the paddings, kernel sizes and strides
        paddings = [paddings] * num_conv_layers if isinstance(paddings, int) else paddings
        kernel_sizes = [kernel_sizes] * num_conv_layers if isinstance(kernel_sizes, int) else kernel_sizes
        strides = [strides] * num_conv_layers if isinstance(strides, int) else strides
        for index, (s, k, p) in enumerate(zip(strides, kernel_sizes, paddings)):
            if s == 1 and isinstance(p, int) and k <= p:
                raise ValueError(f"When using strides=1, the kernel size must be larger than the padding to guarantee "
                                 f"that the output shape is smaller than the input shape. For the {index}-th layer, "
                                 f"the kernel size is {k} and the padding is {p}.")


        self._input_shape = input_shape

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
        dim_analyzer = DimensionsAnalyser(method='static')
        
        # Create a sample input tensor with either provided shape or reasonable default dimensions
        if input_shape is not None:
            sample_input = torch.zeros(1, *input_shape)
            sample_height, sample_width = input_shape[1], input_shape[2]
        else:
            # Use default large dimensions only for trivial kernel sizes (all 1s)
            sample_height, sample_width = 2024, 2024 # setting a very large input here is not issue since the static method does not not pass the input through the torch.nn.Module
            sample_input = torch.zeros(1, channels[0], sample_height, sample_width)
        
        # Get the output shape
        output_shape = dim_analyzer.analyse_dimensions(sample_input.shape, self._block)[1:]
        
        if any(v <= 0 for v in output_shape):
            raise ValueError(f"It seems that the input shape: {self._input_shape} passed to the ResidualConvBlock is completely consumed by the convolutions: {output_shape}. A large input shape is needed !!")


        if output_shape != (channels[0], sample_height, sample_width) or force_residual:
            # Creating an adaptive layer that transforms from input shape to output shape
            self._adaptive_layer = nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[-1],
                kernel_size=(sample_height - output_shape[1] + 1, sample_width - output_shape[2] + 1),
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
        if self._input_shape is not None and x.shape[1:] != self._input_shape:
            raise ValueError(f"Input shape {x.shape[1:]} does not match expected input shape {self._input_shape}")
        
        return self.residual_forward(x, debug=debug)
    
    def __call__(self, x: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.forward(x, debug=debug)
    
    # Override methods with residual implementations
    def children(self) -> Iterator[nn.Module]:
        return super().residual_children()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return super().residual_named_children()
    
    def modules(self) -> Iterator[nn.Module]:
        return super().residual_modules()
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return super().residual_parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        return super().residual_named_parameters(prefix=prefix, recurse=recurse)
    
    def to(self, *args, **kwargs) -> 'ResidualConvBlock':
        super().residual_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'ResidualConvBlock':
        super().residual_train(mode)
        return self
    
    def eval(self) -> 'ResidualConvBlock':
        super().residual_eval()
        return self


    @property
    def adaptive_layer(self) -> Optional[nn.Conv2d]:
        return self._adaptive_layer
    
    @property
    def block(self) -> BasicConvBlock:
        return self._block
    

