"""
This script contains functionality to compute the output dimensions of any given module
either statically or using the module's forward pass
"""
import torch

from torch import nn
from typing import Optional, Union, Tuple
from torch.utils.data import DataLoader

from mypt.code_utils import pytorch_utils as pu
from mypt.dimensions_analysis import layer_specific as lc
from mypt.dimensions_analysis import complex_layers as cl
from torchvision.models.resnet import Bottleneck, BasicBlock

_FORWARD = 'forward_pass'
_STATIC = 'static'

# this constant represents the layer specific types for which a constant time analysis is currently implemented

_DEFAULT_TYPES = (
    # Convolutional layers
    nn.Conv2d,
    nn.ConvTranspose2d,
    
    # Pooling layers
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
    
    # Normalization layers
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm2d,
        
    # Dropout layers
    nn.Dropout,
    nn.Dropout2d,
    
    # Activation layers
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.Sigmoid,
    nn.Tanh,
    
    # Shape manipulation layers
    nn.Flatten,
    nn.Linear,
    nn.Upsample,
    
    # Embedding layer
    nn.Embedding,

    # Complex layers
    Bottleneck
)

_DEFAULT_OUTPUTS = {
    # Convolutional layers
    nn.Conv2d: lc.conv2d_output,
    nn.ConvTranspose2d: lc.convtranspose2d_output,
    
    # Pooling layers
    nn.AvgPool2d: lc.pool2d_output,
    nn.MaxPool2d: lc.pool2d_output,
    nn.AdaptiveMaxPool2d: lc.adaptive_pool2d_output,
    nn.AdaptiveAvgPool2d: lc.adaptive_pool2d_output,
    
    # Normalization layers
    nn.BatchNorm1d: lc.batchnorm1d_output,
    nn.BatchNorm2d: lc.batchnorm2d_output,
    nn.LayerNorm: lc.layernorm_output, 
    nn.GroupNorm: lc.groupnorm_output,
    nn.InstanceNorm2d: lc.instancenorm2d_output,
    
    # Dropout layers  
    nn.Dropout: lc.dropout_output,
    nn.Dropout2d: lc.dropout2d_output,
    
    # Activation layers
    nn.ReLU: lc.activation_output,
    nn.LeakyReLU: lc.activation_output,
    nn.PReLU: lc.activation_output,
    nn.Sigmoid: lc.activation_output,
    nn.Tanh: lc.activation_output,
    
    # Shape manipulation layers
    nn.Flatten: lc.flatten_output,
    nn.Linear: lc.linear_output,
    nn.Upsample: lc.upsample_output,
    
    # Embedding layer
    nn.Embedding: lc.embedding_output,

    # Complex layers
    Bottleneck: cl.resnet_bottleneck_output,
    BasicBlock: cl.resnet_bottleneck_output
}


class DimensionsAnalyser:
    @classmethod
    def analyse_dimensions_forward(cls,
                                   net: nn.Module,
                                   input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int],
                                   device: Optional[str] = None) -> Tuple:
        """
        This function computes the output dimension of the given module, by creating a random tensor,
        executing the module's forward pass and returning the dimensions of the output
        """
        # set the model to the evaluation model
        device = device or pu.get_module_device(net)
        net.eval()
        # make sure the input and the output are on the same device
        input_tensor = torch.ones(*input_shape).to(device)
        output_tensor = net(input_tensor)
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

        if type(net) in _DEFAULT_OUTPUTS:
            return _DEFAULT_OUTPUTS[type(net)](input_shape, net)

        output_shape = input_shape
        # extract the children generator
        children = net.children()

        try:
            first_item = next(children)
            output_shape = cls.analyse_dimensions_static(first_item, output_shape)
        except StopIteration:
            # this means that the generator is empty !!! which breaks the static analysis logic
            raise ValueError(f"The module {net.__class__.__name__} has no children. but is not in the _DEFAULT_TYPES tuple. This detail breaks the static analysis logic")
            # make sure that all simple modules (with no children) are include in the _DEFAULT_TYPES tuple 
            # otherwise, the analysis breaks

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
        return tuple(x.shape) # type: ignore

    @classmethod
    def verify_net(cls,
                   net: nn.Module,
                   input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int, Tuple]) -> bool:
        """
        Verifies that the static analysis output shape matches the forward pass output shape.
        """
        net.eval()
        input_tensor = torch.ones(*input_shape).to(pu.get_module_device(net))
        output_tensor = net(input_tensor)

        out = input_tensor.clone()
        for name, c in net.named_children():
            out = c(out)
        
        if not out.shape == output_tensor.shape:
            raise ValueError(f"Dimension mismatch: Forward pass returned {output_tensor.shape}, but static analysis returned {out.shape}")
        
        return True


    @classmethod
    def debug_dimensions(cls, net: nn.Module, input_shape: Union[lc.three_int_tuple, lc.four_int_tuple, int, Tuple], indent: str = "", device: Optional[str] = None) -> None:
        """
        Recursively finds the exact sub-module where static and forward dimension analysis diverge.
        """
        print(f"{indent}Debugging module: {net.__class__.__name__} with input shape {input_shape}")
        
        try:
            forward_shape = cls.analyse_dimensions_forward(net, input_shape)
        except Exception as e:
            print(f"{indent}[ERROR] Forward pass failed on {net.__class__.__name__} with input shape {input_shape}. Error: {e}")
            return
            
        try:
            static_shape = cls.analyse_dimensions_static(net, input_shape)
        except Exception as e:
            print(f"{indent}[ERROR] Static analysis failed on {net.__class__.__name__} with input shape {input_shape}. Error: {e}")
            return
            
        if isinstance(static_shape, int):
            static_shape = (static_shape,)
            
        if forward_shape == static_shape:
            print(f"{indent}✓ Module {net.__class__.__name__} is consistent. Output shape: {forward_shape}")
            return
            
        print(f"{indent}❌ Divergence found at {net.__class__.__name__}!")
        print(f"{indent}  Forward shape: {forward_shape}")
        print(f"{indent}  Static shape:  {static_shape}")
        
        children = list(net.named_children())
        if not children:
            print(f"{indent}  -> {net.__class__.__name__} is a leaf module and is the root cause of the divergence.")
            return
            
        print(f"{indent}  -> Drilling down into children of {net.__class__.__name__}...")
        current_shape = input_shape
        for name, child in children:
            child_forward = cls.analyse_dimensions_forward(child, current_shape, device=device)
            child_static = cls.analyse_dimensions_static(child, current_shape)
            
            if isinstance(child_static, int):
                child_static = (child_static,)
                
            if child_forward != child_static:
                cls.verify_net(child, current_shape)
                print(f"{indent}  -> Culprit found! Child '{name}' ({child.__class__.__name__}) diverges.")
                cls.debug_dimensions(child, current_shape, indent + "    ", device=device)
                return
                
            current_shape = child_forward

    def __init__(self,
                 net: Optional[nn.Module] = None,
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
                           net: Optional[nn.Module] = None,
                           method: Optional[str] = None):
        if net is None and self.__net is None:
            raise TypeError("Either the 'net' argument or the 'net' field must be passed")

        if net is None:
            net = self.__net

        if method is None:
            method = self.__method

        return self.analyse_dimensions_static(net, input_shape) if method == _STATIC \
            else self.analyse_dimensions_forward(net, input_shape)

