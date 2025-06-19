import torch
import numpy as np

from torch import nn

class NaiveSHA(nn.Module):
    def __init__(self, input_dimension: int, value_dim: int, key_dim: int) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.W_q = nn.Linear(input_dimension, key_dim)  # q_i = W_q * x_i
        self.W_k = nn.Linear(input_dimension, key_dim)  # k_i = W_k * x_i
        self.W_v = nn.Linear(input_dimension, value_dim)  # v_i = W_v * x_i
        
        self.W_o = nn.Linear(value_dim, input_dimension)  # head_i = W_o * v_i

    def _create_attention_mask(self, sequence_length: int) -> torch.Tensor:
        """
        Creates a mask of shape (sequence_length, sequence_length) using explicit loops
        instead of vectorized operations.
        """
        # Create a matrix filled with zeros
        mask = torch.ones(size=(sequence_length, sequence_length))
        
        # Iterate through each row and column
        for i in range(sequence_length):
            for j in range(sequence_length):
                # Set values above the main diagonal to -inf (future tokens)
                if j > i:
                    mask[i, j] = float('-inf')
                # Set values on or below the main diagonal to 1 (current and past tokens)
                else:
                    mask[i, j] = 1.0
                    
        return mask

    def _key_query_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Calculate the dot product between query and key vectors element by element.
        
        Args:
            q: Query tensor of shape (batch_size, sequence_length, key_dim)
            k: Key tensor of shape (batch_size, sequence_length, key_dim)
            
        Returns:
            Dot product tensor of shape (batch_size, sequence_length, sequence_length)
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
        
    def _compute_weights(self, query_key_product: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the attention mask and compute softmax weights with explicit loops.
        
        Args:
            query_key_product: Tensor of shape (batch_size, sequence_length, sequence_length)
            mask: Attention mask of shape (sequence_length, sequence_length)
            
        Returns:
            Attention weights tensor of shape (batch_size, sequence_length, sequence_length)
        """
        batch_size, _, _ = query_key_product.shape
        
        # Initialize the output tensor
        weights = torch.zeros_like(query_key_product)
        
        # Apply mask and compute softmax for each row in each batch
        for b in range(batch_size):
            batch_product = query_key_product[b]

            sign_mask = torch.sign(batch_product) 
            sign_mask += (sign_mask == 0).type(torch.float32)

            # make sure to apply both the mask and the sign mask
            batch_product_masked = batch_product * sign_mask * mask 
            
            # apply softmax
            weights[b] = torch.softmax(batch_product_masked, dim=-1)

        return weights

    def _compute_new_v(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of values with explicit loops.
        
        Args:
            weights: Attention weights tensor of shape (batch_size, sequence_length, sequence_length)
            v: Value tensor of shape (batch_size, sequence_length, value_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, value_dim)
        """
        batch_size, sequence_length, _ = weights.shape
        
        # Initialize output tensor
        output = torch.zeros(batch_size, sequence_length, self.value_dim)
        
        # Compute weighted sum for each position in each batch
        for b in range(batch_size):
            for i in range(sequence_length):
                output[b, i, :] = weights[b, i, :] @ v[b, :, :]
                 

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input using naive loops instead of vectorized operations.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dimension)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, input_dimension)
        """
        if x.ndim != 3:
            raise ValueError(f"The input tensor is expected to be of the shape (batch_size, sequence_length, input_dimension), but got {x.shape}")

        batch_size, sequence_length, __ = x.shape 

        # Apply linear transformations to get query, key, and value
        q = self.W_q.forward(x)  # (batch_size, sequence_length, key_dim)
        k = self.W_k.forward(x)  # (batch_size, sequence_length, key_dim)
        v = self.W_v.forward(x)  # (batch_size, sequence_length, value_dim)

        # Compute attention scores
        query_key_product = self._key_query_product(q, k)
        
        # Create attention mask
        mask = self._create_attention_mask(sequence_length)
        
        # Compute attention weights
        weights = self._compute_weights(query_key_product, mask)
        
        # Compute weighted sum of values
        output = self._compute_new_v(weights, v)
        
        # Apply output projection
        final_output = self.W_o.forward(output)
        
        return final_output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def children(self) -> list[nn.Module]:
        return [self.W_q, self.W_k, self.W_v, self.W_o]

    def named_children(self) -> list[tuple[str, nn.Module]]:
        return [("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v), ("W_o", self.W_o)]

    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    def parameters(self):
        for child in self.children():
            yield from child.parameters()
