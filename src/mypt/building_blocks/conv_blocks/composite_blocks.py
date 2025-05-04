import torch
from torch import nn
from typing import List, Tuple, Union

from mypt.building_blocks.mixins.general import SequentialModuleListMixin
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin
from mypt.building_blocks.conv_blocks.conv_block_design.contracting_designer import ContractingCbDesigner


class ContractingBlock(WrapperLikeModuleMixin, SequentialModuleListMixin):
    """
    A convolutional block that contracts the spatial dimensions of the input.
    """

    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 output_shape: Tuple[int, int, int],
                 max_conv_layers_per_block: int = 4,
                 min_conv_layers_per_block: int = 2,
                 inner_model_field_name: str = '_block',
                 *args, **kwargs):
        WrapperLikeModuleMixin.__init__(self, inner_model_field_name)
        SequentialModuleListMixin.__init__(self, inner_model_field_name)
        self.blocks = ContractingCbDesigner(input_shape, output_shape, max_conv_layers_per_block, min_conv_layers_per_block).get_contracting_block()
        self._block = nn.ModuleList(self.blocks)


    def to(self, *args, **kwargs) -> 'ContractingBlock':
        return super().module_list_to(*args, **kwargs)
        
    def train(self, mode: bool = True) -> 'ContractingBlock':
        return super().module_list_train(mode)
    
    def eval(self) -> 'ContractingBlock':
        return super().module_list_eval()
    

    def _full_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        final_output = x

        for block in self.blocks:
            final_output = block.conv(final_output)
            outputs.append(final_output)
            final_output = block.pool(final_output)

        return outputs
        

    def forward(self, x: torch.Tensor, full: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        output = super().sequential_module_list_forward(x)

        if full:
            intermediate_outputs = self._full_forward(x)
            return output, intermediate_outputs

        return output
        

