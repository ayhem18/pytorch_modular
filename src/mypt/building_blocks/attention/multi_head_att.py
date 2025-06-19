"""
This module is my attempt to efficiently implement the multi-head attention layer: Using a single linear layer to compute the query, key and value matrics.
"""

import torch

import numpy as np

from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int) -> None:
        super().__init__()

        if (d_model % num_heads != 0):
            raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}")


        self.d_model = d_model
        self.num_heads = num_heads


        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(d_model, key_dim * num_heads)
        self.W_k = nn.Linear(d_model, key_dim * num_heads)
        self.W_v = nn.Linear(d_model, value_dim * num_heads)

        self.W_o = nn.Linear(value_dim * num_heads, d_model) 

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
        batch_size, _, sequence_length, _ = q.shape 

        query_key_product = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.key_dim).item()

        # compute the dot product of the query and key matrices 
        # query_key_product = torch.bmm(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.key_dim).item()
        
        # the result must be of the shape (batch_size, num_heads, sequence_length, sequence_length)
        if query_key_product.shape != (batch_size, self.num_heads, sequence_length, sequence_length):
            raise ValueError(f"The shape of the query_key_product tensor is expected to be (batch_size, num_heads, sequence_length, sequence_length), but got {query_key_product.shape}")

        return query_key_product



    def _compute_weights(self, query_key_product: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        """
        # the query_key_product must be of the shape (batch_size, num_heads, sequence_length, sequence_length)

        batch_size, _, sequence_length, _ = query_key_product.shape 
        
        # so funny enough, neg_value * (-inf) = +inf, which means the product has mixed inf values
        # messing up the softmax operation

        # so I need to create a mask with the signs of the query_key_product while mapping any zeros to 1 (so that 0 * -inf = -inf (masked zeros))
        sign_mask = torch.sign(query_key_product) 
        sign_mask += (sign_mask == 0).type(torch.float32)

        weights = torch.softmax(query_key_product * sign_mask * mask, dim=-1)

        if weights.shape != (batch_size, self.num_heads, sequence_length, sequence_length):
            raise ValueError(f"The shape of the weights tensor is expected to be (batch_size, num_heads, sequence_length, sequence_length), but got {weights.shape}")

        return weights


    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, _, sequence_length, _ = v.shape

        output = torch.matmul(weights, v)
        
        # the output must be of the shape (batch_size, num_heads, sequence_length, value_dim)
        if output.shape != (batch_size, self.num_heads, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, num_heads, sequence_length, value_dim), but got {output.shape}")

        return output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        batch_size, sequence_length, __ = x.shape 
    
        q = self.W_q.forward(x)
        k = self.W_k.forward(x)
        v = self.W_v.forward(x)

        # create a view for efficiency
        # it is easier to work with (batch_size, num_heads, sequence_length, key_dim)
        q = q.view(batch_size, sequence_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, sequence_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, sequence_length, self.num_heads, self.value_dim).permute(0, 2, 1, 3)

        query_key_product = self._key_query_product(q, k)

        mask = self._create_attention_mask(sequence_length).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1).to(q.device)

        weights = self._compute_weights(query_key_product, mask)

        new_v = self._compute_new_v(weights, v)

        # the new_v must be of the shape (batch_size, num_heads, sequence_length, value_dim)
        if new_v.shape != (batch_size, self.num_heads, sequence_length, self.value_dim):
            raise ValueError(f"The shape of the new_v tensor is expected to be (batch_size, num_heads, sequence_length, value_dim), but got {new_v.shape}")

        output = self.W_o.forward(new_v.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, self.num_heads * self.value_dim))

        if output.shape != (batch_size, sequence_length, self.d_model):
            raise ValueError(f"The shape of the output tensor is expected to be (batch_size, sequence_length, d_model), but got {output.shape}")

        return output


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def children(self) -> list[nn.Module]:
        return [self.W_q, self.W_k, self.W_v, self.W_o] 


    def named_children(self) -> list[tuple[str, nn.Module]]:
        return [("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v), ("W_o", self.W_o)] 
    

    def train(self, mode: bool = True) -> "MultiHeadAttentionLayer":    
        """
        """
        super().train(mode)
        for c in self.children():
            c.train(mode)
        return self
    
    def eval(self) -> "MultiHeadAttentionLayer":    
        """
        """
        return self.train(False)
    
    def to(self, *args, **kwargs) -> "MultiHeadAttentionLayer":
        """
        """
        for c in self.children():
            c.to(*args, **kwargs)
        return self
    
