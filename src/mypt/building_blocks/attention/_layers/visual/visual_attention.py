"""
VisualAttentionBlock
--------------------
A lightweight building block that converts a 2-D feature map into a sequence of
patch tokens, adds positional information, applies **one** multi-head
self-attention layer, and reshapes the sequence back to a feature map.

Unlike the `VisualTransformerBlock` (which internally contains LayerNorms,
residual connections, and an FFN), this block delegates **only** the attention
operation to an `AbstractMHAttentionLayer` instance that is supplied from the
outside.  No extra layer-norms, residuals, or MLPs are added here, making this
component suitable for experimenting with attention-only augmentations inside
CNN/U-Net pipelines or diffusion models.
"""

from typing import Tuple, Optional

import torch
from torch import nn

# type: ignore because mypy might not see dynamic package structure
from mypt.building_blocks.attention.multi_head_att import AbstractMHAttentionLayer  # type: ignore
from .vis_patcher import (
    VisBasicPatcher,
    VisConvPatcher,
    VisUnfoldPatcher,
    AbstractPatcher,
)
from .vis_embedding import (
    AbstractVisualEmbedding,
    LearnableVisualEmbedding,
    FixedSinCos2DVisualEmbedding,
    FourierFeatures2DVisualEmbedding,
)


class VisualAttentionBlock(nn.Module):
    """Attention-only visual block: patch → embed → attention → un-patch."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        patch_size: int,
        attention_layer: AbstractMHAttentionLayer,
        *,
        patcher_type: str = "unfold",
        embedding_type: str = "learnable",
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.patch_size = patch_size

        # ------------------------------------------------------------------
        # Patcher
        # ------------------------------------------------------------------
        self.patcher: AbstractPatcher = self._build_patcher(
            patcher_type, patch_size, input_shape
        )

        # d_model inferred from patcher output
        self.embedding_dim = self.patcher.output_dim
        self.num_patches = self.patcher.num_patches

        # Sanity-check compatibility with the provided attention layer
        if getattr(attention_layer, "d_model", None) is not None and attention_layer.d_model != self.embedding_dim:
            raise ValueError(
                "Attention layer d_model ({} ) does not match patcher output_dim ({}).".format(
                    attention_layer.d_model, self.embedding_dim
                )
            )

        # ------------------------------------------------------------------
        # Positional Embedding
        # ------------------------------------------------------------------
        self.embedding: AbstractVisualEmbedding = self._build_embedding(
            embedding_type, self.patcher, self.embedding_dim
        )

        # ------------------------------------------------------------------
        # Attention layer (supplied externally)
        # ------------------------------------------------------------------
        self.attention = attention_layer

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _build_patcher(
        self, patcher_type: str, patch_size: int, input_shape: Tuple[int, int, int]
    ) -> AbstractPatcher:
        if patcher_type == "basic":
            return VisBasicPatcher(patch_size=patch_size, input_shape=input_shape)
        if patcher_type == "conv":
            return VisConvPatcher(patch_size=patch_size, input_shape=input_shape)
        if patcher_type == "unfold":
            return VisUnfoldPatcher(patch_size=patch_size, input_shape=input_shape)
        raise ValueError(f"Unknown patcher_type: {patcher_type}")

    def _build_embedding(
        self, embedding_type: str, patcher: AbstractPatcher, embedding_dim: int
    ) -> AbstractVisualEmbedding:
        # derive patch grid size
        num_patches_h = self.input_shape[1] // patcher.patch_size
        num_patches_w = self.input_shape[2] // patcher.patch_size

        if embedding_type == "learnable":
            return LearnableVisualEmbedding(patcher.num_patches, embedding_dim)
        if embedding_type == "sincos":
            return FixedSinCos2DVisualEmbedding(
                num_patches_h, num_patches_w, embedding_dim
            )
        if embedding_type == "fourier":
            return FourierFeatures2DVisualEmbedding(
                num_patches_h, num_patches_w, embedding_dim
            )
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    def _unpatch(self, seq: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, D) back to (B, D, H', W')."""
        b, n, d = seq.shape
        h_p = self.input_shape[1] // self.patch_size
        w_p = self.input_shape[2] // self.patch_size
        if n != h_p * w_p:
            raise ValueError("Sequence length does not match expected patch grid size.")
        seq = seq.view(b, h_p, w_p, d).permute(0, 3, 1, 2).contiguous()
        return seq

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401, E501
        # 1. Flatten into patch tokens
        patches = self.patcher(x)  # (B, N, D)

        # 2. Add positional encodings
        tokens = self.embedding(patches)  # (B, N, D)

        # 3. Self-attention (no extra LayerNorm/residuals inside this block)
        tokens_out = self.attention(tokens, mask) if mask is not None else self.attention(tokens)

        # 4. Reconstruct 2-D feature map
        return self._unpatch(tokens_out)
