import torch
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
    """
    This is a simple implementation of the position encoding originally introduced in the "Attention is all you need" paper.

    Args:
        dim_embdding (int): The dimension of the embedding.
        max_period (int, optional): The maximum period of the positional encoding. Defaults to 10000.
    """

    def __init__(self, 
                dim_embedding:int,
                max_period: int = 10000):
        super().__init__()

        # the embedding will always be made even
        self.dim_embedding = dim_embedding + int(dim_embedding % 2 == 1)
        self.max_period = max_period

        # according to the paper, for a timestampt 't' and a dimension 'i' : 
        # (here it seems reasonable to assume that i goes from 0 to dim_embedding // 2)
        # PE(pos, 2i) = sin(pos / max_period^(2i/dim_embedding))
        # PE(pos, 2i+1) = cos(pos / max_period^(2i/dim_embedding))

        # the input is a tensor of shape (batch_size,)
        # so we need to create a tensor of shape (dim_embedding,) where all even elements are equal to 1 / max_period^(2i)

        self.sin_positions = (torch.arange(0, self.dim_embedding, 2) / self.dim_embedding)
        self.cos_positions = (torch.arange(1, self.dim_embedding, 2) / self.dim_embedding)

        # for numerical stability express 1 / max_period ^ (2i/dim_embedding) as exp(-log(max_period) * (2i/dim_embedding))        
        self.sin_positions = torch.exp(self.sin_positions * - torch.log(torch.tensor(self.max_period)))[None, :]
        self.cos_positions = torch.exp(self.cos_positions * - torch.log(torch.tensor(self.max_period)))[None, :]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in [1, 2]:
            raise ValueError(f"Input tensor must be 1D or 2D, got {x.ndim}D")
        
        if x.ndim == 2 and x.shape[1] != 1:
            raise ValueError(f"The input tensor is expected to be of shape (batch_size,) or (batch_size, 1), got {x.shape}")        
        
        # expand the input tensor to the shape of the positional encoding table
        if x.ndim == 1:
            x = x[:, None] # this converts the input from shape (batch_size,) to shape (batch_size, 1)
        
        # create a pe_table of shape (batch_size, dim_embedding)
        pe_table = torch.zeros(x.shape[0], self.dim_embedding, device=x.device)

        # set the values of the table
        pe_table[:, 0::2] = torch.sin(self.sin_positions.to(x.device) * x)
        pe_table[:, 1::2] = torch.cos(self.cos_positions.to(x.device) * x)

        return pe_table


class GaussianFourierEncoding(nn.Module):
    """
    This is a simple implementation of the Gaussian Fourier encoding 
    Args:
        embedding_dim (int): The dimension of the embedding.
        scale (float, optional): The scale of the weights. Defaults to 1.0.
        use_log (bool, optional): Whether to use the log of the input. Defaults to True.
    """

    def __init__(self, 
            embedding_dim: int = 256, 
            scale: float = 1.0, 
            use_log: bool = True
    ):
        super().__init__()
        self.embedding_dim = 2 * embedding_dim
        
        # random non-learnable weights
        self.weight = nn.Parameter(torch.randn(embedding_dim) * scale, requires_grad=False)
        self.use_log = use_log


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # few sanity checks to avoid broadcasting unexpected silent bugs
        if x.ndim not in [1, 2]:
            raise ValueError(f"Input tensor must be 1D or 2D, got {x.ndim}D")
        
        if x.ndim == 2 and x.shape[1] != 1:
            raise ValueError(f"The input tensor is expected to be of shape (batch_size,) or (batch_size, 1), got {x.shape}")    
        
        # expand the input tensor to shape (batch_size, 1)
        if x.ndim == 1:
            x = x[:, None]
                
        if self.use_log:
            x = torch.log(x)

        # broadcase the weight from (embedding_dim,) to (batch_size, embedding_dim)
        x_proj = x * self.weight[None, :] * 2 * np.pi

        # concatenate the cos and sin projections
        out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return out
