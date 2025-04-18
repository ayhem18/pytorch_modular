import torch

from mypt.building_blocks.mixins.general import ModuleListMixin
from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.linear_blocks.components import ResidualFullyConnectedBlock


class MultiResidualFCBlock(ModuleListMixin):
    """
    This class implements a fully connected block with multiple inner fully connected blocks inside.
    """
    def _verify_instance(self):
        attrs = ['layers_per_residual_block', 'units', 'activation', 'dropout']
        for att in attrs:
            if not hasattr(self, att):
                raise AttributeError(f"The ResidualMixin expects the child class to include the following attributes: {attrs}. Found {att} missing")            

        if self.layers_per_residual_block <= 1:
            raise ValueError(f"Residual blocks must contain at least 2 layers. Found: {self.layers_per_residual_block} layer(s) per residual block")
        


    def _build(self) -> torch.nn.ModuleList:

        num_layers = len(self.units) - 1

        if num_layers % self.layers_per_residual_block == 0:
            residual_range = num_layers - self.layers_per_residual_block
        else:
            residual_range = (num_layers // self.layers_per_residual_block) * self.layers_per_residual_block

        blocks = [
            ResidualFullyConnectedBlock(output=self.units[i + self.layers_per_residual_block],
                                in_features=self.units[i],
                                units=self.units[i: i + self.layers_per_residual_block + 1],
                                num_layers=self.layers_per_residual_block,
                                dropout=self.dropout[i: i + self.layers_per_residual_block],
                                activation=self.activation
                                )
            for i in range(0, residual_range, self.layers_per_residual_block)
        ]

        # add the last block as a standard Fully connected block
        blocks.append(GenericFCBlock(in_features=self.units[residual_range + 1], 
                                    output=self.units[-1], 
                                    num_layers=num_layers - residual_range, 
                                    units=self.units[residual_range:],
                                    dropout=self.dropout[residual_range:],
                                    activation=self.activation,
                                    )
                    )
        
        return torch.nn.ModuleList(blocks)