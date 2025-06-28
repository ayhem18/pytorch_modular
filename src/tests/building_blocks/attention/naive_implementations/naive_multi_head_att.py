import abc
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from typing import Optional


class AbstractNaiveMHA(nn.Module, abc.ABC):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int) -> None:
        """
        Naive implementation of Multi-Head Attention for testing purposes.
        
        Args:
            d_model (int): Model dimension (input and output dimension)
            num_heads (int): Number of attention heads
            value_dim (int): Dimension of value vectors for each head
            key_dim (int): Dimension of key/query vectors for each head
        """
        super().__init__()
        
        if (d_model % num_heads != 0):
            raise ValueError(f"d_model must be divisible by num_heads, but got d_model={d_model} and num_heads={num_heads}")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_dim = key_dim
        
        # Initialize Q, K, V projection matrices for each head separately (non-vectorized approach)
        self.W_q_list = nn.ModuleList([nn.Linear(d_model, key_dim) for _ in range(num_heads)])
        self.W_k_list = nn.ModuleList([nn.Linear(d_model, key_dim) for _ in range(num_heads)])
        self.W_v_list = nn.ModuleList([nn.Linear(d_model, value_dim) for _ in range(num_heads)])
        
        # Output projection
        self.W_o = nn.Linear(value_dim * num_heads, d_model)


    @abc.abstractmethod
    def _default_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        pass


    def create_final_mask(self, causal_mask: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pad_mask is None:
            return causal_mask
        pad_bool = pad_mask.to(dtype=torch.bool)
        b, s = pad_bool.shape
        row = pad_bool.unsqueeze(-1).unsqueeze(1).expand(b, self.num_heads, s, s)
        col = pad_bool.unsqueeze(1).unsqueeze(1).expand(b, self.num_heads, s, s)
        return causal_mask & row & col


    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute query-key product for a single head without vectorization.
        
        Args:
            q: Query tensor [batch_size, seq_len, key_dim]
            k: Key tensor [batch_size, seq_len, key_dim]
            
        Returns:
            Product tensor [batch_size, seq_len, seq_len]
        """
        batch_size, sequence_length, _ = q.shape
        
        # Initialize the output tensor
        result = torch.zeros(batch_size, sequence_length, sequence_length)
        
        # Iterate through each batch
        for b in range(batch_size):
            # Iterate through each query position
            for i in range(sequence_length):
                # Iterate through each key position
                for j in range(sequence_length):
                    # Calculate the dot product between q[b,i] and k[b,j]
                    dot_product = (torch.dot(q[b, i], k[b, j])).item()

                    # Scale by square root of key dimension
                    result[b, i, j] = dot_product / np.sqrt(self.key_dim).item()
        
        return result

    
    def _compute_weights(self, query_key_product: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the boolean attention mask and compute softmax weights with explicit loops.
        
        Args:
            query_key_product: Tensor of shape (B, H, S, S)
            final_mask: Boolean attention mask of shape (B, H, S, S), where True means keep.
            
        Returns:
            Attention weights tensor of shape (B, H, S, S)
        """        


        batch_size, sequence_length, _ = query_key_product.shape
        
        weights = torch.zeros_like(query_key_product)

        for b in range(batch_size):
            for i in range(sequence_length):
                row_scores = query_key_product[b, i]
                row_mask = final_mask[b, i]
                
                masked_scores = row_scores[row_mask]
                
                if masked_scores.numel() > 0:
                    softmax_scores = F.softmax(masked_scores, dim=0)
                    weights[b, i, row_mask] = softmax_scores

        return weights


    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted sum of values without vectorization.
        
        Args:
            weights: Attention weights [batch_size, seq_len, seq_len]
            v: Value tensor [batch_size, seq_len, value_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, value_dim]
        """
        batch_size, sequence_length, _ = weights.shape
        
        # Initialize output tensor
        output = torch.zeros(batch_size, sequence_length, self.value_dim)
        
        # Compute weighted sum for each position in each batch
        for b in range(batch_size):
            for i in range(sequence_length):
                output[b, i, :] = weights[b, i, :] @ v[b, :, :]
                 

        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using explicit loops instead of vectorized operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """

        # since this implementation is for testing purposes only, I do not move the vectors to the same device
        batch_size, seq_len, _ = x.shape
        
        # Create attention mask
        causal_mask = self._default_mask(batch_size, seq_len).to(x.device)
        final_mask = self.create_final_mask(causal_mask, mask.to(x.device) if mask is not None else None)
        
        # Process each head separately and collect outputs
        head_outputs = [None for _ in range(self.num_heads)]
        
        for h in range(self.num_heads):
            # Project input for this head
            q_head = self.W_q_list[h](x)
            k_head = self.W_k_list[h](x)
            v_head = self.W_v_list[h](x)
            
            # Compute attention scores
            qk_product = self._key_query_product(q_head, k_head)
            
            # Apply masking and compute weights
            attn_weights = self._compute_weights(qk_product, final_mask[:, h])

            # Compute weighted sum of values
            head_output = self._compute_new_v(attn_weights, v_head)
            
            # Add to list of head outputs
            head_outputs[h] = head_output
        
        # Concatenate head outputs manually
        concatenated = torch.zeros(batch_size, seq_len, self.num_heads * self.value_dim)
        for h in range(self.num_heads):
            for b in range(batch_size):
                for i in range(seq_len):
                    for d in range(self.value_dim):
                        concatenated[b, i, h * self.value_dim + d] = head_outputs[h][b, i, d]
        
        # Apply output projection
        output = self.W_o(concatenated)
        
        return output
    
    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, mask)
        
    # Helper methods for syncing weights with vectorized implementation
    def sync_weights_from_vectorized(self, vectorized_mha):
        """
        Synchronize weights with the vectorized implementation for testing.
        This extracts slices from the vectorized model's weights.
        
        Args:
            vectorized_mha: The vectorized MultiHeadAttentionLayer instance
        """
        # Extract and set weights for each head
        for h in range(self.num_heads):
            # For W_q
            self.W_q_list[h].weight.data = vectorized_mha.W_q.weight[h*self.key_dim:(h+1)*self.key_dim].clone()
            self.W_q_list[h].bias.data = vectorized_mha.W_q.bias[h*self.key_dim:(h+1)*self.key_dim].clone()
            
            # For W_k
            self.W_k_list[h].weight.data = vectorized_mha.W_k.weight[h*self.key_dim:(h+1)*self.key_dim].clone()
            self.W_k_list[h].bias.data = vectorized_mha.W_k.bias[h*self.key_dim:(h+1)*self.key_dim].clone()
            
            # For W_v
            self.W_v_list[h].weight.data = vectorized_mha.W_v.weight[h*self.value_dim:(h+1)*self.value_dim].clone()
            self.W_v_list[h].bias.data = vectorized_mha.W_v.bias[h*self.value_dim:(h+1)*self.value_dim].clone()
        
        # For W_o
        self.W_o.weight.data = vectorized_mha.W_o.weight.clone()
        self.W_o.bias.data = vectorized_mha.W_o.bias.clone()




class CausalNaiveMHA(AbstractNaiveMHA):
    
    def _default_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        causal = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
        for i in range(sequence_length):
            for j in range(sequence_length):
                causal[i, j] = j <= i
        return causal.unsqueeze(0).unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)


class BidirectionalNaiveMHA(AbstractNaiveMHA):
    def _default_mask(self, batch_size: int, sequence_length: int) -> torch.Tensor:
        return torch.ones(batch_size, self.num_heads, sequence_length, sequence_length, dtype=torch.bool) # all-ones mask

