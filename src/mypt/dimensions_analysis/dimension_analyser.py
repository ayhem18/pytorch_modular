"""
This script contains functionality to compute the output dimensions of any given module
either statically or using the module's forward pass
"""
import torch

from torch import nn
from typing import Union, Tuple
from torch.utils.data import DataLoader

from ..code_utilities.pytorch_utilities import get_module_device
from ..dimensions_analysis import layer_specific as lc

_FORWARD = 'forward_pass'
_STATIC = 'static'

# this constant represents the types for which the
_DEFAULT_TYPES = (nn.Conv2d,
                  nn.AvgPool2d,
                  nn.MaxPool2d,
                  nn.AdaptiveAvgPool2d,
                  nn.AdaptiveMaxPool2d,
                  nn.Flatten,
                  nn.Linear)

_DEFAULT_OUTPUTS = {
    nn.Conv2d: lc.conv2d_output,
    nn.AvgPool2d: lc.pool2d_output,
    nn.MaxPool2d: lc.pool2d_output,
    nn.AdaptiveMaxPool2d: lc.adaptive_pool2d_output,
    nn.AdaptiveAvgPool2d: lc.adaptive_pool2d_output,
    nn.Flatten: lc.flatten_output,
    nn.Linear: lc.linear_output
}


class DimensionsAnalyser:
    @classmethod
    def analyse_dimensions_forward(cls,
                                   net: nn.Module,
                                   input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int]) -> Tuple:
        """
        This function computes the output dimension of the given module, by creating a random tensor,
        executing the module's forward pass and returning the dimensions of the output
        """
        module_device = get_module_device(net)
        # make sure the input and the output are on the same device
        input_tensor = torch.ones(*input_shape).to(module_device)
        # set the model to the evaluation model
        net.eval()
        output_tensor = net.forward(input_tensor)
        return tuple(output_tensor.size())

    @classmethod
    def analyse_dimensions_static(cls,
                                  net: nn.Module,
                                  input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int]) -> Union[Tuple, int]:
        """
        This function computes the output dimension of the given module when passing a tensor of the given input shape
        by recursively iterating through the module's layers.

        NOTE:
        This function might return unreliable output if the module's architecture does not align with its forward logic
        """

        # first base case: if module is simply a default layer
        if isinstance(net, _DEFAULT_TYPES):
            return _DEFAULT_OUTPUTS[type(net)](input_shape, net)

        output_shape = input_shape
        # extract the children generator
        children = net.children()
        # if the generator is empty, then the module does not change the dimensions of the input
        for child in children:
            output_shape = cls.analyse_dimensions_static(child, output_shape)

        return output_shape

    @classmethod
    def analyse_dimensions_dataloader(cls, dataloader: DataLoader) -> Tuple[int, int, int, int]:
        # first convert to an iterator
        batch = next(iter(dataloader))
        # if the data loader returns a tuple, then it is usually batch of image and a batch of labels
        x = batch if isinstance(batch, tuple) else batch[0]
        return tuple(x.shape)

    def __init__(self,
                 net: nn.Module = None,
                 method: str = _FORWARD):
        """
        The constructor sets the forward pass as the default method as it is more reliable.
        The module's architecture can be different from the forward pass logic. The latter governs
        the output's dimensions.
        """

        self.__net = net
        self.__method = method

    def analyse_dimensions(self,
                           input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int],
                           net: nn.Module = None,
                           method: str = None):
        if net is None and self.__net is None:
            raise TypeError("Either the 'net' argument or the 'net' field must be passed")

        if net is None:
            net = self.__net

        if method is None:
            method = self.__method

        return self.analyse_dimensions_static(net, input_shape) if method == _STATIC \
            else self.analyse_dimensions_forward(net, input_shape)

