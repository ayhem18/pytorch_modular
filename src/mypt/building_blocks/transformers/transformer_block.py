import abc
import torch

from typing import Optional

from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.attention.multi_head_att import AbstractMHAttentionLayer, CausalMultiHeadAttentionLayer, BidirectionalMultiHeadAttentionLayer



class AbstractTransformerBlock(NonSequentialModuleMixin, torch.nn.Module, abc.ABC):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        torch.nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['ln1', 'att', 'ln2', 'ffn'])

        self.d_model = d_model
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.dropout = dropout


        self.att = self._set_attention()

        self.ffn = GenericFCBlock(output=d_model,
                                  in_features=d_model,
                                  num_layers=2,
                                  units=[d_model, d_model * 4, d_model],
                                  activation='gelu',
                                  dropout=dropout,
                                  # make sure to pass the "norm_layer" argument since the default is "batchnorm1d" which wouldn't work with sequence inputs
                                  norm_layer='layernorm' 
                                  )

        self.ln1 = torch.nn.LayerNorm(normalized_shape=(d_model,))
        self.ln2 = torch.nn.LayerNorm(normalized_shape=(d_model,))


    @abc.abstractmethod
    def _set_attention(self) -> AbstractMHAttentionLayer:
        pass

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # I am implementing the forward pass as explained in the stanford NLP book: 
        # chapter 9, section 2 
        residual = x
        x = self.ln1(x)
        x = self.att(x, pad_mask)
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x + residual 
    

    def __call__(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, pad_mask)
    


class CausalTransformerBlock(AbstractTransformerBlock):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        super().__init__(d_model, num_heads, value_dim, key_dim, dropout)

    def _set_attention(self) -> CausalMultiHeadAttentionLayer: 
        return CausalMultiHeadAttentionLayer(self.d_model, self.num_heads, self.value_dim, self.key_dim)
    


class BidirectionalTransformerBlock(AbstractTransformerBlock):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        super().__init__(d_model, num_heads, value_dim, key_dim, dropout)

    def _set_attention(self) -> BidirectionalMultiHeadAttentionLayer:
        return BidirectionalMultiHeadAttentionLayer(self.d_model, self.num_heads, self.value_dim, self.key_dim)