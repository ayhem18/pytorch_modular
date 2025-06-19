"""
This file contains my implementation of the attention layer, according to the thorough explanation provided in the 9th chapter of the this amazing Standford resource: 
https://web.stanford.edu/~jurafsky/slp3/ 
"""

import torch 
import numpy as np

from torch import nn
from typing import Generator


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, input_dimension: int, value_dim: int, key_dim: int) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(input_dimension, key_dim) # q_i = W_q * x_i
        self.W_k = nn.Linear(input_dimension, key_dim) # k_i = W_k * x_i
        self.W_v = nn.Linear(input_dimension, value_dim) # v_i = W_v * x_i
        
        self.W_o = nn.Linear(value_dim, input_dimension) # head_i = W_o * v_i


    def _create_attention_mask(self, sequence_length: int) -> torch.Tensor:
        """
        Creates a mask of shape (sequence_length, sequence_length) with the lower triangular part set to 1 and the upper triangular part set to 0
        """
        # the mask must satisfy the following conditions: 
        # of the shape (sequence_length, sequence_length)
        # the lower triangular part (including the diagonal) must be 1 and the upper triangular part must be -inf

        # create the ones part
        ones_mask = torch.tril(torch.ones(size=(sequence_length, sequence_length)), diagonal=0) 

        # create the inf part 
        inf_mask = torch.ones(size=(sequence_length, sequence_length)) * float("-inf") 
        
        inf_mask = torch.tril(inf_mask, diagonal=-1).T 

        # combine the two masks

        mask = inf_mask + ones_mask

        return mask

    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = q.shape 

        # compute the dot product of the query and key matrices
        # the result must be of the shape (batch_size, sequence_length, sequence_length) 
        query_key_product = torch.bmm(q, k.permute(0, 2, 1)) / np.sqrt(self.key_dim).item()
        
        if query_key_product.shape != (batch_size, sequence_length, sequence_length):
            raise ValueError(f"The shape of the query_key_product tensor is expected to be (batch_size, sequence_length, sequence_length), but got {query_key_product.shape}")

        return query_key_product

    def _compute_weights(self, query_key_product: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = query_key_product.shape 
        
        # so funny enough, neg_value * (-inf) = +inf, which means the product has mixed inf values
        # messing up the softmax operation

        # so I need to create a mask with the signs of the query_key_product while mapping any zeros to 1 (so that 0 * -inf = -inf (masked zeros))
        sign_mask = torch.sign(query_key_product) 
        sign_mask += (sign_mask == 0).type(torch.float32)

        weights = torch.softmax(query_key_product * sign_mask * mask, dim=2)

        if weights.shape != (batch_size, sequence_length, sequence_length):
            raise ValueError(f"The shape of the weights tensor is expected to be (batch_size, sequence_length, sequence_length), but got {weights.shape}")

        return weights

    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = v.shape

        output = torch.bmm(weights, v)
        
        # the output must be of the shape (batch_size, sequence_length, value_dim)
        if output.shape != (batch_size, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, sequence_length, value_dim), but got {output.shape}")

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be of the shape (batch_size, sequence_length, input_dimension)
        """

        if x.ndim != 3:
            raise ValueError(f"The input tensor is expected to be of the shape (batch_size, sequence_length, input_dimension), but got {x.shape}")

        batch_size, sequence_length, __ = x.shape 

        q = self.W_q.forward(x) # (batch_size, sequence_length, key_dim)
        k = self.W_k.forward(x) # (batch_size, sequence_length, key_dim)
        v = self.W_v.forward(x) # (batch_size, sequence_length, value_dim)

        query_key_product = self._key_query_product(q, k)

        mask = self._create_attention_mask(sequence_length).unsqueeze(0).expand(batch_size, -1, -1)

        weights = self._compute_weights(query_key_product, mask)
        
        output = self._compute_new_v(weights, v) 

        return self.W_o.forward(output)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def children(self) -> list[nn.Module]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def named_children(self) -> list[tuple[str, nn.Module]]:
        return [("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v), ("W_o", self.W_o)]


    def modules(self) -> Generator[nn.Module]:
        # yield the actual module as per Pytorch conventions
        yield self

        for child in self.children():
            yield from child.modules()


    def train(self, mode: bool = True) -> "SingleHeadAttentionLayer":
        """
        """
        super().train(mode)
        for c in self.children():
            c.train(mode)
        return self

    def eval(self) -> "SingleHeadAttentionLayer":
        """
        """
        return self.train(False)
    
    def to(self, *args, **kwargs) -> "SingleHeadAttentionLayer":
        """
        """
        for c in self.children():
            c.to(*args, **kwargs)
        return self
    