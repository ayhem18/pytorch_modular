"""
This script contains the implementation of the functionalities shared among the One Dimensional and Three Dimensional CondResnetBlocks. 

The functionalities shared are:
- the conditional residual block
- the conditional residual block with spatial downsampling
- the conditional residual block with spatial upsampling

"""

import torch

from torch import nn
from abc import abstractmethod, ABC
from typing import Callable, List, Optional, Tuple, Union

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.mixins.residual_mixins import GeneralResidualMixin
from mypt.building_blocks.auxiliary.normalization.utils import _normalization_functions


class AbstractCondResnetBlock(NonSequentialModuleMixin, GeneralResidualMixin, nn.Module, ABC):
    """
    Abstract class for the conditional residual blocks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    _WIDE_RESNET_BLOCK_NORMALIZATION_FUNCTIONS = {
        "groupnorm": nn.GroupNorm,
        "batchnorm": nn.BatchNorm2d,
    }

    def __validate_normalization_function(self, 
                                          norm: Optional[Union[nn.Module, str]] = None,
                                          norm_params: Optional[dict] = None) -> Tuple[nn.Module, dict]:
        # TODO: MAKE SURE TO UPDATE THIS METHOD AS MORE NORMALIZATION FUNCTIONS ARE ADDED TO THE _normalization_functions DICTIONARY !!!

        if isinstance(norm, str):
            if norm not in self._WIDE_RESNET_BLOCK_NORMALIZATION_FUNCTIONS:
                raise NotImplementedError(f"Normalization type {norm} is not currently supported !!!")
            # convert norm to a callable object
            norm = _normalization_functions.get(norm)

        norm = norm or nn.GroupNorm # the default normalization is groupnorm 
        norm_params = norm_params or {}

        return norm, norm_params


    def __set_num_groups(self, norm_group_params: dict, num_channels: int) -> int:
        """
        Set the number of groups for the GroupNorm layer.
        """

    
        if "num_groups" in norm_group_params:
            return norm_group_params["num_groups"]

        if num_channels % 2 == 1:
            return 1
        
        max_power_2 = 1

        while num_channels % (2 ** max_power_2) != 0 and max_power_2 < 5:
            max_power_2 += 1

        return min(2 ** max_power_2, num_channels // 4)

    def __set_norm_params(self, 
                          film_block: int, 
                          norm: Union[nn.Module, str], 
                          norm_params: Optional[dict]) -> Tuple[nn.Module, dict]:
        
        norm, norm_params = self.__validate_normalization_function(norm, norm_params)

        channels = self._in_channels if film_block == 1 else self._out_channels

        # at this point, norm is a Callable 
        if norm == nn.GroupNorm:
            norm_params["num_channels"] = channels
            norm_params["num_groups"] = self.__set_num_groups(norm_params, channels)
        
        elif norm == nn.BatchNorm2d:
            norm_params["num_features"] = norm_params.get("num_features", channels)
        
        return norm, norm_params 
    

    def __set_residual_stream(self) -> Tuple[str, str]:
        # can be a one-liner, but let's focus on clarify for now
        if not self._force_residual and (self._stride == 1 and self._in_channels == self._out_channels):
            # the shortcut connection can be a simple identity if the stride is 1 and the input and output channels are the same 
            residual_stream_field_name = None

        else:
            residual_stream_field_name = "_shortcut"

        # Create shortcut connection if needed
        if residual_stream_field_name is not None:
            self._shortcut = nn.Conv2d(
                self._in_channels, 
                self._out_channels, 
                kernel_size=3, 
                stride=self._stride, 
                padding=1,
            )
        else:
            self._shortcut = None

        return "_components", residual_stream_field_name


    @abstractmethod
    def set_components(self) -> nn.ModuleDict:
        pass


    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
        
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

        # whether to have a convolutional layer as a shortcut connection when the input and output shapes are identical
        force_residual: bool = False, 

        extra_components_fields: Optional[List[str]] = None,
    ):
        # Validate the `stride` parameter
        if stride not in [1, 2]:
            raise ValueError(f"Stride must be 1 or 2, got {stride}")
       
        # Initialize the nn.Module
        nn.Module.__init__(self)
       
        # Initialize the NonSequentialModuleMixin
        NonSequentialModuleMixin.__init__(
            self,
            inner_components_fields=[
                "_components",  
                "_shortcut"
            ] + (extra_components_fields or []) 
        )
        
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_dimension = cond_dimension

        self._stride = stride
        self._dropout_rate = dropout_rate  
        self._force_residual = force_residual

        # set the normalization parameters
        norm1, norm1_params = self.__set_norm_params(1, norm1, norm1_params)
        norm2, norm2_params = self.__set_norm_params(2, norm2, norm2_params)

        # set the default values for the activation parameters (using ReLu until I learn more about activation functions and their properties)
        activation = activation or nn.ReLU
        activation_params = activation_params or {}


        self._film_params1 = {
            "out_channels": in_channels,
            "cond_dimension": cond_dimension,

            "normalization": norm1,
            "normalization_params": norm1_params,
            "activation": activation,
            "activation_params": activation_params,
        
            "inner_dim": inner_dim,
            "film_activation": film_activation,
            "film_activation_params": film_activation_params
        }

        self._film_params2 = {            
            "out_channels": out_channels,
            "cond_dimension": cond_dimension,

            "normalization": norm2,
            "normalization_params": norm2_params,
            "activation": activation,
            "activation_params": activation_params,

            "inner_dim": inner_dim,
            "film_activation": film_activation,
            "film_activation_params": film_activation_params,
        }

        self._components = self.set_components()

        main_stream_field_name, residual_stream_field_name = self.__set_residual_stream()

        # initialize the GeneralResidualMixin
        GeneralResidualMixin.__init__(
            self,
            main_stream_field_name=main_stream_field_name,
            residual_stream_field_name=residual_stream_field_name  
        )


    @abstractmethod
    def _forward_main_stream(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # this must be overriden by the subsequent classes
        pass


    def forward(self, x: torch.Tensor, condition: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # to pass both the `debug` and `condition` parameters to the GeneralResidualMixin.residual_forward() method,
        # debug must be passed as a positional argument and condition must be passed as an extra argument (as part of the `*args` argument)
        return super().residual_forward(x, debug, condition)


    def __call__(self, x: torch.Tensor, condition: torch.Tensor, debug: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Support for optional debug parameter in __call__"""
        return self.forward(x, condition, debug=debug)
    
    # the NonSequentialModuleMixin implementation will take precedence over the torch.nn.Module implementation for the `to, train, eval...` methods !!


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
    def cond_dimension(self) -> int:
        """the number of features in the conditioning tensor"""
        return self._cond_dimension
    
    @property
    def stride(self) -> int:
        """Returns the stride used in the block."""
        return self._stride
    
    @property
    def dropout_rate(self) -> float:
        """Returns the dropout rate used in the block."""
        return self._dropout_rate


class AbstractCondUpWResBlock(NonSequentialModuleMixin, nn.Module):
    """
    Abstract class for the conditional upsampling residual blocks.
    """

    def __init__(self,
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
        upsample_type: str):

        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=["_upsample", "_resnet_block"])
                
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_dimension = cond_dimension
        self._upsample_type = upsample_type

        # the self._resnet_block must be initialized by sub-classe
        self._resnet_block: AbstractCondResnetBlock = None

    
        # Create the upsampling layer
        self._upsample = self._create_upsample_layer(
            upsample_type=upsample_type,
        )


    def _create_upsample_layer(
        self,
        upsample_type: str,
    ) -> nn.Module:
        
        """
        Creates the upsampling layer based on the specified type.
        
        for interpolation and upsampling, the mode is set to 'nearest-exact' because of the note in the pytorch documentation: 
        https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html 

    
        """
        
        if upsample_type == "transpose_conv":
            # the tranpose convolutional layer upsample the input by a factor of 2
            # every single parameter below was chosen carefully to ensure that h_out = h_in * 2 (regardless of h_in!!) 
            return nn.ConvTranspose2d(
                in_channels=self._out_channels, # the convolutional layer will receive the output of the resnet block (in_channels of upsample layer is out_channels of resnet block)
                out_channels=self._out_channels,
                kernel_size=3,
                stride=2,
                padding=1, 
                output_padding=1 
            )
        elif upsample_type == "conv":
            return nn.Sequential(
                # upsample
                nn.Upsample(scale_factor=2, mode='nearest-exact'),
                # apply a convolutional layer that does not change the dimensions
                nn.Conv2d(
                    in_channels=self._out_channels, # the convolutional layer will receive the output of the upsample layer (in_channels of upsample layer is out_channels of resnet block)
                    out_channels=self._out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
        elif upsample_type == "interpolate":
            return nn.Upsample(scale_factor=2, mode='nearest-exact')
        else:
            raise ValueError(f"Unknown upsample type: {upsample_type}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling conditioned resnet block."""
        x = self._resnet_block(x, condition)
        x = self._upsample(x)
        return x


class AbstractCondDownWResBlock(NonSequentialModuleMixin, nn.Module):
    """
    Abstract class for the conditional downsampling residual blocks.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dimension: int,
        downsample_type: str):

        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=["_downsample", "_resnet_block"])
        
        # Store parameters
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._cond_dimension = cond_dimension
        self._downsample_type = downsample_type
        
        # Create the conditional resnet block
        self._resnet_block: AbstractCondResnetBlock = None

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
                self._out_channels, # the convolutional layer will receive the output of the resnet block (in_channels of downsample layer is out_channels of resnet block)
                self._out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        elif downsample_type == "avg_pool":
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    self._out_channels, # the convolutional layer will receive the output of the resnet block (in_channels of downsample layer is out_channels of resnet block)
                    self._out_channels,
                    kernel_size=1,
                    stride=1, 
                    padding=0
                )
            )
        elif downsample_type == "max_pool":
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    self._out_channels,
                    self._out_channels,
                    kernel_size=1,
                    stride=1, 
                    padding=0
                )
            )
        else:
            raise ValueError(f"Unknown downsample type: {downsample_type}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling conditioned resnet block."""
        x = self._resnet_block(x, condition)
        x = self._downsample(x)
        return x
