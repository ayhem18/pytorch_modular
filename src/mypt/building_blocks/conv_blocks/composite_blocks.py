import torch

from torch import nn
from typing import Iterator, List, Tuple, Union

from mypt.building_blocks.mixins.general import SequentialModuleListMixin
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin
from mypt.building_blocks.conv_blocks.conv_block_design.expanding_designer import ExpandingCbDesigner
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
        super().module_list_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'ContractingBlock':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'ContractingBlock':
        super().module_list_eval()
        return self

    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()


    def _full_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_output = x

        outputs = [None for _ in range(len(self.blocks))]

        for block_index, block in enumerate(self.blocks):
            final_output = block.conv(final_output)
            outputs[block_index] = final_output
            final_output = block.pool(final_output)

        return outputs
        

    def forward(self, x: torch.Tensor, full: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        output = super().sequential_module_list_forward(x)

        if full:
            intermediate_outputs = self._full_forward(x)
            return output, intermediate_outputs

        return output
        

    @property
    def block(self) -> List[nn.Module]:
        return self._block


class ExpandingBlock(WrapperLikeModuleMixin, SequentialModuleListMixin):
    """
    A transpose convolutional block that expands the spatial dimensions of the input.
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
        self.blocks = ExpandingCbDesigner(input_shape, output_shape, max_conv_layers_per_block, min_conv_layers_per_block).get_expanding_block()
        self._block = nn.ModuleList(self.blocks)


    def to(self, *args, **kwargs) -> 'ExpandingBlock':
        super().module_list_to(*args, **kwargs)
        return self
    
    def train(self, mode: bool = True) -> 'ExpandingBlock':
        super().module_list_train(mode)
        return self
    
    def eval(self) -> 'ExpandingBlock':
        super().module_list_eval()
        return self

    def modules(self) -> Iterator[nn.Module]:
        return super().module_list_modules()


    def _full_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_output = x

        outputs = [None for _ in range(len(self.blocks))]

        # In expanding blocks, we'd just record the output after each block
        for block_index, block in enumerate(self.blocks):
            final_output = block(final_output)
            outputs[block_index] = final_output
            
        return outputs
        

    def forward(self, x: torch.Tensor, full: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        output = super().sequential_module_list_forward(x)

        if full:
            intermediate_outputs = self._full_forward(x)
            return output, intermediate_outputs

        return output
        

    @property
    def block(self) -> List[nn.Module]:
        return self._block


