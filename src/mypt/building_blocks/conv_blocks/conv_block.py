# the main idea here is to create a convolutional block that can be used to build larger networks.

import torch
from torch import nn
from typing import List, Optional, OrderedDict, Tuple, Union

from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin


class BasicConvBlock(WrapperLikeModuleMixin):
    """
    A flexible convolutional block that can be configured with multiple layers.
    
    This class allows creating a sequence of convolutional layers with optional 
    batch normalization and activation functions.
    """
    
    def __build_block_single_activation(self, 
                                        channels: List[int], 
                                        kernel_sizes: List[int], 
                                        stride: List[int], 
                                        padding: List[int], 
                                        use_bn: bool, 
                                        activation: nn.Module, 
                                        activation_params: dict):
        # define the activation layer 
        activation_layer = activation(**activation_params) if activation_params else activation()
        layers = []
        for i in range(len(channels) - 1):
            layers.append((f"conv_{i + 1}", nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], stride[i], padding[i])))
            if use_bn:
                layers.append((f"bn_{i + 1}", nn.BatchNorm2d(channels[i+1])))

        # at the very end, add the activation layer 
        layers.append((f"activation_layer", activation_layer))
        return nn.Sequential(OrderedDict(layers))


    def __build_block_activation_after_each_layer(self, 
                                        channels: List[int], 
                                        kernel_sizes: List[int], 
                                        stride: List[int], 
                                        padding: List[int], 
                                        use_bn: bool, 
                                        activation: nn.Module, 
                                        activation_params: dict):
        # define the activation layer 
        layers = []

        for i in range(len(channels) - 1):
            # add the convolutional layer
            layers.append((f"conv_{i + 1}", nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], stride[i], padding[i])))
            # add the batch normalization layer
            if use_bn:
                layers.append((f"bn_{i + 1}", nn.BatchNorm2d(channels[i+1])))
            # add the activation layer
            # create a new activation layer each time (it seems that the Sequential module does not support having the same object in the list multiple times)
            activation_layer = activation(**activation_params) if activation_params else activation()
            layers.append((f"activation_{i + 1}", activation_layer))
        
        return nn.Sequential(OrderedDict(layers))


    def __init__(self, 
                 num_conv_layers: int, 
                 channels: Union[List[int], Tuple[int]],
                 kernel_sizes: Union[List[int], int],
                 strides: Optional[Union[List[int], int]] = 1,
                 paddings: Optional[Union[List[int], List[str], str, int]] = 'same',
                 use_bn:bool=True,
                 activation_after_each_layer: bool = True,
                 activation: Optional[nn.Module] = None,
                 activation_params: Optional[dict] = None,
                ):
        """The block can be designed in two ways: 

        [conv layer, bn, activation, conv layer, bn, activation ...]
        or 
        [conv layer, bn, conv layer, bn, ... conv layer, bn, activation ]

        Args:
            num_conv_layers (int): _description_
            channels (Union[List[int], Tuple[int]]): _description_
            kernel_sizes (Union[List[int], int]): _description_
            stride (Optional[Union[List[int], int]], optional): _description_. Defaults to 1.
            padding (Optional[Union[List[int], List[str], str, int]], optional): _description_. Defaults to 'same'.
            use_bn (bool, optional): _description_. Defaults to True.
            activation (Optional[nn.Module], optional): _description_. Defaults to None.
            activation_params (Optional[dict], optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        # the BasicConvBlock is a wrapper around a nn.Sequential module saved in the '_block' field
        super().__init__("_block")

        if not isinstance(channels, Union[List, Tuple]):
            raise TypeError(f"The 'channels' argument must be a list or a tuple. Found: {type(channels)}")
        
        # make sure the number of channels is the number of conv layers + 1        
        if len(channels) != num_conv_layers + 1:
            raise ValueError(f"The number of channels must be the number of conv layers + 1. Found: {len(channels)} channels for {num_conv_layers} conv layers.")
        
        # convert all the other parameters to lists if needed
        if not isinstance(kernel_sizes, Union[List, Tuple]):
            kernel_sizes = [kernel_sizes] * num_conv_layers
        else:
            if len(kernel_sizes) != num_conv_layers:
                raise ValueError(f"The number of kernel sizes must be the number of conv layers. Found: {len(kernel_sizes)} kernel sizes for {num_conv_layers} conv layers.")


        if not isinstance(strides, Union[List, Tuple]):
            strides = [strides] * num_conv_layers
        else:
            if len(strides) != num_conv_layers:
                raise ValueError(f"The number of strides must be the number of conv layers. Found: {len(strides)} strides for {num_conv_layers} conv layers.")

        if not isinstance(paddings, Union[List, Tuple]):
            paddings = [paddings] * num_conv_layers
        else:
            if len(paddings) != num_conv_layers:
                raise ValueError(f"The number of paddings must be the number of conv layers. Found: {len(paddings)} paddings for {num_conv_layers} conv layers.")

        # set the fields
        self._num_conv_layers = num_conv_layers  
        self._channels = channels
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        self._use_bn = use_bn
        self._activation_after_each_layer = activation_after_each_layer

        if activation is None:
            activation = nn.ReLU

        if activation_params is None:
            activation_params = {}

        self._activation = activation
        self._activation_params = activation_params

        if activation_after_each_layer:
            self._block = self.__build_block_activation_after_each_layer(channels, kernel_sizes, strides, paddings, use_bn, activation, activation_params) 
        else:
            self._block = self.__build_block_single_activation(channels, kernel_sizes, strides, paddings, use_bn, activation, activation_params) 

    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # all the necessary methods are inherited from the WrapperLikeModuleMixin
    # train(), eval(), to(), forward(), __str__(), __repr__(), etc...   

    # add properties for the fields 
    @property
    def num_conv_layers(self) -> int:
        return self._num_conv_layers
    
    @property
    def channels(self) -> List[int]:
        return self._channels  
    
    @property
    def kernel_sizes(self) -> List[int]:
        return self._kernel_sizes 
    
    @property
    def block(self) -> nn.Sequential:
        return self._block
    