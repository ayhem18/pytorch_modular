"""
This module provides various strategies for creating positional embeddings for
sequences of visual patches, as used in Vision Transformers and other
attention-based visual models.

The goal is to provide positional information to a model that otherwise
treats the input as an unordered set of tokens.
"""

import torch

from torch import nn
from abc import ABC, abstractmethod

from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding, GaussianFourierEncoding


class AbstractVisualEmbedding(nn.Module, ABC):
    """
    Abstract base class for all visual positional embedding modules.
    It defines the common interface for adding positional information to a
    tensor of patch embeddings.
    """
    def __init__(self,
                 num_patches: int,
                 embedding_dim: int
                 ):
        super().__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional embedding to the input tensor.

        Args:
            x (torch.Tensor): The input patch embeddings of shape (B, N, D),
                              where N is num_patches and D is embedding_dim.

        Returns:
            torch.Tensor: The patch embeddings with added positional information.
        """
        pass


class LearnableVisualEmbedding(AbstractVisualEmbedding):
    """
    Creates a learnable positional embedding for each patch.

    This is the standard approach used in the original Vision Transformer (ViT).
    An embedding vector is learned for every possible patch position.

    References:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
          (https://arxiv.org/abs/2010.11929)
    """
    def __init__(self, num_patches: int, embedding_dim: int):
        super().__init__(num_patches, embedding_dim)
        # Create a learnable parameter for the positional embeddings.
        # Shape: (1, num_patches, embedding_dim) for easy broadcasting.
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the learned positional embeddings to the patch embeddings."""
        if x.shape[1] != self.num_patches:
            raise ValueError(f"Input tensor has {x.shape[1]} patches, but expected {self.num_patches}.")
        return x + self.positional_embedding


class FixedSinCos2DVisualEmbedding(AbstractVisualEmbedding):
    """
    Creates fixed 2D positional embeddings using sine and cosine functions.

    This extends the 1D positional encoding from "Attention Is All You Need" to
    two dimensions. Separate 1D sinusoidal embeddings are generated for the
    height and width coordinates and then concatenated.
    """
    def __init__(self, num_patches_h: int, num_patches_w: int, embedding_dim: int):
        num_patches = num_patches_h * num_patches_w
        super().__init__(num_patches, embedding_dim)

        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but got {embedding_dim}.")

        pos_enc_dim = embedding_dim // 2
        self.pos_enc_h = PositionalEncoding(dim_embedding=pos_enc_dim)
        self.pos_enc_w = PositionalEncoding(dim_embedding=pos_enc_dim)

        # Create positional encodings for height and width
        h_positions = torch.arange(num_patches_h, dtype=torch.float32)
        w_positions = torch.arange(num_patches_w, dtype=torch.float32)

        h_emb = self.pos_enc_h(h_positions)  # (num_patches_h, D/2)
        w_emb = self.pos_enc_w(w_positions)  # (num_patches_w, D/2)

        # Combine to create a 2D grid
        h_emb_grid = h_emb.unsqueeze(1).repeat(1, num_patches_w, 1) # (H, W, D/2)
        w_emb_grid = w_emb.unsqueeze(0).repeat(num_patches_h, 1, 1) # (H, W, D/2)

        combined_emb = torch.cat([h_emb_grid, w_emb_grid], dim=-1) # (H, W, D)

        # Flatten and register as a non-learnable buffer
        flat_emb = combined_emb.view(1, self.num_patches, self.embedding_dim)
        self.register_buffer('positional_embedding', flat_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the pre-computed 2D sinusoidal embeddings."""
        if x.shape[1] != self.num_patches:
            raise ValueError(f"Input tensor has {x.shape[1]} patches, but expected {self.num_patches}.")
        return x + self.positional_embedding


class FourierFeatures2DVisualEmbedding(AbstractVisualEmbedding):
    """
    Creates fixed 2D positional embeddings using Gaussian Fourier Features.

    This method maps the 2D coordinates of each patch to a higher-dimensional
    space using random Fourier features. This can capture high-frequency
    details in the positional information.
    """
    def __init__(self, num_patches_h: int, num_patches_w: int, embedding_dim: int, scale: float = 10.0):
        num_patches = num_patches_h * num_patches_w
        super().__init__(num_patches, embedding_dim)

        if embedding_dim % 4 != 0:
            raise ValueError(f"Embedding dimension must be divisible by 4, but got {embedding_dim}.")

        # GaussianFourierEncoding outputs 2 * input_dim, so we target half
        fourier_enc_dim = embedding_dim // 2
        self.fourier_enc_h = GaussianFourierEncoding(embedding_dim=fourier_enc_dim, scale=scale, use_log=False)
        self.fourier_enc_w = GaussianFourierEncoding(embedding_dim=fourier_enc_dim, scale=scale, use_log=False)

        # Create normalized coordinates for height and width in [-1, 1]
        h_coords = torch.linspace(-1.0, 1.0, num_patches_h)
        w_coords = torch.linspace(-1.0, 1.0, num_patches_w)

        h_emb = self.fourier_enc_h(h_coords) # (num_patches_h, D/2)
        w_emb = self.fourier_enc_w(w_coords) # (num_patches_w, D/2)

        # Combine to create a 2D grid
        h_emb_grid = h_emb.unsqueeze(1).repeat(1, num_patches_w, 1)
        w_emb_grid = w_emb.unsqueeze(0).repeat(num_patches_h, 1, 1)

        combined_emb = torch.cat([h_emb_grid, w_emb_grid], dim=-1)

        # Flatten and register as a non-learnable buffer
        flat_emb = combined_emb.view(1, self.num_patches, self.embedding_dim)
        self.register_buffer('positional_embedding', flat_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the pre-computed 2D Fourier feature embeddings."""
        if x.shape[1] != self.num_patches:
            raise ValueError(f"Input tensor has {x.shape[1]} patches, but expected {self.num_patches}.")
        return x + self.positional_embedding 