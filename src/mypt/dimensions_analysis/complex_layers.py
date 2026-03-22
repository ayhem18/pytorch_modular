from torch import nn
from typing import Tuple, Union

from torchvision.models.resnet import BasicBlock, Bottleneck

def resnet_bottleneck_output(input_shape: Union[Tuple, int], block: Union[BasicBlock, Bottleneck, nn.Module]) -> Union[Tuple, int]:
    """
    Computes the output shape of a ResNet Bottleneck block.
    It follows the main sequential branch and ignores the parallel 'downsample' branch.
    """
    from mypt.dimensions_analysis.dimension_analyser import DimensionsAnalyser
    
    if not isinstance(block, (BasicBlock, Bottleneck)):
        raise TypeError(f"The block is not a BasicBlock or Bottleneck. Got {type(block)}")

    output_shape = input_shape
    for name, child in block.named_children():
        if name == 'downsample':
            continue
        # Recursively call the static analyser on the children of the main branch
        output_shape = DimensionsAnalyser.analyse_dimensions_static(child, output_shape)
    return output_shape
