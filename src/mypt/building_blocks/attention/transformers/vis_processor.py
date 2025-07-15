"""
This module prepare a 4 dimensional tensor to be passed into the attention block. The visualAttentionInputProcessor is an integral part of the VisualAttentionBlock.
The processor consists of 

1. a patcher that converts the (bs, c, h, w) tensor into a (bs, n_patches, d_model) tensor 
2. a positional encoder that adds positional information to the patches 
3. 

"""
import torch

from torch import nn
from typing import Tuple, Optional

from mypt.building_blocks.attention.layers.visual.vis_patcher import (
    VisBasicPatcher,
    VisConvPatcher,
    VisUnfoldPatcher,
    AbstractPatcher,
)

from mypt.building_blocks.attention.layers.visual.vis_embedding import (
    AbstractVisualEmbedding,
    LearnableVisualEmbedding,
    FixedSinCos2DVisualEmbedding,
    FourierFeatures2DVisualEmbedding,
)

from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.building_blocks.attention.transformers.transformer_block import AbstractTransformerBlock


class VisualTransformerBlock(NonSequentialModuleMixin):
    """
    A full visual attention block, similar to a Transformer Encoder block.
    It processes a 2D feature map by patching it, adding positional embeddings,
    applying multi-head self-attention, passing it through an MLP, and then
    un-patching it back to a 2D feature map.

    This module is designed to be a drop-in replacement for a convolutional
    block within a larger architecture like a U-Net.
    """

    def _get_patcher(self, patcher_type: str, patch_size: int, input_shape: Tuple[int, int, int]) -> AbstractPatcher:
        if patcher_type == 'basic':
            return VisBasicPatcher(patch_size=patch_size, input_shape=input_shape)
        elif patcher_type == 'conv':
            return VisConvPatcher(patch_size=patch_size, input_shape=input_shape)
        elif patcher_type == 'unfold':
            return VisUnfoldPatcher(patch_size=patch_size, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown patcher_type: {patcher_type}")

    def _get_embedding(self, embedding_type: str, patcher: AbstractPatcher, embedding_dim: int) -> AbstractVisualEmbedding:
        num_patches_h = self.input_shape[1] // self.patcher.patch_size
        num_patches_w = self.input_shape[2] // self.patcher.patch_size

        if embedding_type == 'learnable':
            return LearnableVisualEmbedding(num_patches=self.patcher.num_patches, embedding_dim=embedding_dim)
        elif embedding_type == 'sincos':
            return FixedSinCos2DVisualEmbedding(num_patches_h=num_patches_h, num_patches_w=num_patches_w, embedding_dim=embedding_dim)
        elif embedding_type == 'fourier':
            return FourierFeatures2DVisualEmbedding(num_patches_h=num_patches_h, num_patches_w=num_patches_w, embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")



    def __init__(self,
                 input_shape: Tuple[int, int, int],  # (C, H, W)
                 patch_size: int,
                 transformer_block: AbstractTransformerBlock,
                 patcher_type: str = 'unfold',
                 embedding_type: str = 'learnable',
                 ):
        super().__init__(inner_components_fields=["patcher", "embedding", "transformer"])
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.patcher: AbstractPatcher = self._get_patcher(patcher_type, patch_size, input_shape)

        self.embedding_dim = self.patcher.output_dim
        self.num_patches = self.patcher.num_patches

        if self.embedding_dim != transformer_block.d_model:
            raise ValueError(
                f"Patcher output dimension ({self.embedding_dim}) must match transformer d_model ({transformer_block.d_model})"
            )

        self.embedding: AbstractVisualEmbedding = self._get_embedding(
            embedding_type, self.patcher, self.embedding_dim
        )

        # External transformer block that already contains LN, MHSA, FFN, and residuals
        self.transformer = transformer_block


    def _unpatch(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes the sequence of patches back to a 2D feature map."""
        B, N, D = x.shape
        H_patch = self.input_shape[1] // self.patch_size
        W_patch = self.input_shape[2] // self.patch_size
        
        # (B, N, D) -> (B, H_patch * W_patch, D)
        if N != H_patch * W_patch:
            raise ValueError("Number of patches in tensor does not match expected H_patch * W_patch.")
            
        # (B, H_patch * W_patch, D) -> (B, H_patch, W_patch, D)
        x = x.view(B, H_patch, W_patch, D)
        
        # (B, H_patch, W_patch, D) -> (B, D, H_patch, W_patch)
        x = x.permute(0, 3, 1, 2)
        
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Patch and add positional embedding
        patches = self.patcher(x)  # (B, N, D)
        patches_with_pos = self.embedding(patches)  # (B, N, D)

        # 2. Pass through the provided transformer block
        seq_out = self.transformer(patches_with_pos, pad_mask=mask)

        # 3. Unpatch back to feature map
        output_map = self._unpatch(seq_out)

        return output_map
        
        
class VisualTransformerBlockCond(NonSequentialModuleMixin):
    def __init__(self, 
                ):
        super().__init__(inner_components_fields=["patcher", "embedding", "transformer"])
        
        
        