import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.auxiliary.film_block import TwoDimFiLMBlock, ThreeDimFiLMBlock


class ConditionalWResBlock(NonSequentialModuleMixin):
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
    
    def __initialize_film_block(self, 
                                normalization: Union[str, Callable],
                                normalization_params: dict,
                                activation: Union[str, Callable],
                                activation_params: dict,
                                out_channels: int,
                                cond_channels: int,
                                film_dimension: int,
                                inner_dim: int,
                                film_activation: Union[str, Callable],
                                film_activation_params: dict,
                                ) -> Union[TwoDimFiLMBlock, ThreeDimFiLMBlock]:
        
        if film_dimension == 2:
            return TwoDimFiLMBlock(
                normalization=normalization,
                normalization_params=normalization_params,
                activation=activation,
                activation_params=activation_params,
                out_channels=out_channels,
                cond_channels=cond_channels,
                inner_dim=inner_dim,
                film_activation=film_activation,
                film_activation_params=film_activation_params
            )
        elif film_dimension == 3:
            return ThreeDimFiLMBlock(
                normalization=normalization,
                normalization_params=normalization_params,
                activation=activation,
                activation_params=activation_params,
                out_channels=out_channels,
                cond_channels=cond_channels,
                inner_dim=inner_dim,
                film_activation=film_activation,
                film_activation_params=film_activation_params
            )
        
        raise ValueError(f"Invalid film dimension: {film_dimension}")

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        
        film_dimension: int,
        inner_dim: int = 256,
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
        force_residual: bool = False, 
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
        self._film_dimension = film_dimension
        self._inner_dim = inner_dim
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
        self._components = nn.ModuleDict()

        self._components["film1"] = self.__initialize_film_block(
            normalization=norm1,
            normalization_params=norm1_params,
            activation=activation,
            activation_params=activation_params,
            out_channels=in_channels,
            cond_channels=cond_channels,
            film_dimension=film_dimension,
            inner_dim=inner_dim,
            film_activation=film_activation,
            film_activation_params=film_activation_params
            )
        
        self._components["conv1"] = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
        )
        
        self._components["dropout"] = nn.Dropout(dropout_rate)

        self._components["film2"] = self.__initialize_film_block(
            normalization=norm2,
            normalization_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            out_channels=out_channels,
            cond_channels=cond_channels,
            film_dimension=film_dimension,
            inner_dim=inner_dim,
            film_activation=film_activation,
            film_activation_params=film_activation_params
        )
        
        self._components["conv2"] = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1,  # Always 1 for the second conv
            padding=1, 
        )
        
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
    
    # the other methods are inherited from the NonSequentialModuleMixin 

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


class UpCondWResnetBlock(NonSequentialModuleMixin):
    """
    A conditioned WideResnet block with upsampling capabilities.
    The block passes the input through a ConditionalWResBlock and then applies upsampling.
    
    Supports multiple upsampling methods:
    - transpose_conv: Uses ConvTranspose2d
    - conv: Uses Conv2d after interpolation
    - interpolate: Uses only interpolation
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        film_dimension: int,
        inner_dim: int = 256,
        stride: int = 1,
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

        upsample_type: str = "transpose_conv",
    ):
        super().__init__(inner_components_fields=["_upsample", "_resnet_block"])
                
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_channels = cond_channels
        self._upsample_type = upsample_type


        # Create the conditional resnet block
        self._resnet_block = ConditionalWResBlock(
            in_channels=in_channels,  
            out_channels=out_channels,
            cond_channels=cond_channels,
            film_dimension=film_dimension,
            inner_dim=inner_dim,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual
        )
        
    
        # Create the upsampling layer
        self._upsample = self._create_upsample_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_type=upsample_type,
        )


    def _create_upsample_layer(
        self,
        in_channels: int,
        out_channels: int,
        upsample_type: str,
    ) -> nn.Module:
        
        """Creates the upsampling layer based on the specified type."""
        if upsample_type == "transpose_conv":
            # the tranpose convolutional layer upsample the input by a factor of 2
            return nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        elif upsample_type == "conv":
            return nn.Sequential(
                # upsample
                nn.Upsample(scale_factor=2, mode='nearest'),
                # apply a convolutional layer that does not change the dimensions
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
        elif upsample_type == "interpolate":
            return nn.Upsample(scale_factor=2, mode='nearest')
        else:
            raise ValueError(f"Unknown upsample type: {upsample_type}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling conditioned resnet block."""
        x = self._resnet_block(x, condition)
        x = self._upsample(x)
        return x


class DownCondWResnetBlock(NonSequentialModuleMixin):
    """
    A conditioned WideResnet block with downsampling capabilities.
    The block passes the input through a ConditionalWResBlock and then applies downsampling.
    
    Supports multiple downsampling methods:
    - conv: Uses strided convolution
    - avg_pool: Uses average pooling
    - max_pool: Uses max pooling
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        film_dimension: int,
        inner_dim: int = 256,
        stride: int = 1,
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
        downsample_type: str = "conv",
    ):
        super().__init__(inner_components_fields=["_downsample", "_resnet_block"])
        
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_channels = cond_channels
        self._downsample_type = downsample_type

        
        # Create the conditional resnet block
        self._resnet_block = ConditionalWResBlock(
            in_channels=out_channels,  # Input channels is now out_channels after downsampling
            out_channels=out_channels,
            cond_channels=cond_channels,
            film_dimension=film_dimension,
            inner_dim=inner_dim,
            stride=stride,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual
        )

        # Create the downsampling layer
        self._downsample = self._create_downsample_layer(
            downsample_type=downsample_type,
        )


    
    def _create_downsample_layer(
        self,
        downsample_type: str,
    ) -> nn.Module:
        
        """Creates the downsampling layer based on the specified type."""
        if downsample_type == "conv":
            return nn.Conv2d(
                self._in_channels,
                self._out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        elif downsample_type == "avg_pool":
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    self._out_channels,
                    self._out_channels,
                    kernel_size=1,
                    stride=1
                )
            )
        elif downsample_type == "max_pool":
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    self._out_channels,
                    self._out_channels,
                    kernel_size=1,
                    stride=1
                )
            )
        else:
            raise ValueError(f"Unknown downsample type: {downsample_type}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling conditioned resnet block."""
        x = self._resnet_block(x, condition)
        x = self._downsample(x)
        return x
