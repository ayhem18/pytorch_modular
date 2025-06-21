import torch

from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.attention.multi_head_att import MultiHeadAttentionLayer


class TransformerBlock(NonSequentialModuleMixin, torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        torch.nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['ln1', 'att', 'ln2', 'ffn'])

        self.d_model = d_model
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.att = MultiHeadAttentionLayer(d_model, num_heads, value_dim, key_dim)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I am implementing the forward pass as explained in the stanford NLP book: 
        # chapter 9, section 2 
        residual = x
        x = self.ln1(x)
        x = self.att(x)
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x + residual 
    

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
