import torch

from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.attention.multi_head_att import MultiHeadAttentionLayer


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

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
                                  dropout=dropout)

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)

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
    
    def children(self) -> list[torch.nn.Module]:
        return [self.att, self.ffn, self.ln1, self.ln2]
    
    def named_children(self) -> list[tuple[str, torch.nn.Module]]:
        return [('att', self.att), ('ffn', self.ffn), ('ln1', self.ln1), ('ln2', self.ln2)]

    def train(self, mode: bool = True) -> "TransformerBlock":
        self.att.train(mode)
        self.ffn.train(mode)
        self.ln1.train(mode)
        self.ln2.train(mode)
        return self
    
    def eval(self) -> "TransformerBlock":
        self.train(False)
        return self

    def to(self, *args, **kwargs) -> "TransformerBlock":
        self.att.to(*args, **kwargs)
        self.ffn.to(*args, **kwargs)
        self.ln1.to(*args, **kwargs)
        self.ln2.to(*args, **kwargs)
        return self


