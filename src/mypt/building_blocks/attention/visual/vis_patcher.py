"""
This module is responsbile for patching 4 dimensional tensor into a 3 dimensional input. 

The idea here is to have (B, C, H, W) converted into (B, N, D) where N and D are computed as follows: 

given a patch size of P, N = (H/P) * (W/P) and D = C * (P * P)
""" 

import torch

from typing import Tuple
from abc import ABC, abstractmethod

from mypt.building_blocks.conv_blocks.basic.conv_block import BasicConvBlock
from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin


class AbstractPatcher(torch.nn.Module, ABC):
    def __init__(self, 
                patch_size: int,
                input_shape: Tuple[int, int, int]
                ):
        super().__init__()
        # do the checks 
        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(f"The input shape must be divisible by the patch size. Got {input_shape} and patch size {patch_size}")

        self.patch_size = patch_size
        self.input_shape = input_shape
        # N: number of patches
        self.num_patches = (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        # D: output dimension
        self.output_dim = input_shape[0] * patch_size * patch_size

    @abstractmethod 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



class VisBasicPatcher(AbstractPatcher):
    # this module inherits from nn.Module because it does not have an inner model.
    def __init__(self, 
                patch_size: int,
                input_shape: Tuple[int, int, int]
                ):        
        super().__init__(patch_size, input_shape)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size
        n_h = H // P
        n_w = W // P

        # Reshape to (B, C, n_h, P, n_w, P)
        x = x.view(B, C, n_h, P, n_w, P)
        # Permute to (B, n_h, n_w, C, P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # Reshape to (B, n_h * n_w, C * P * P) -> (B, num_patches, output_dim)
        x = x.contiguous().view(B, self.num_patches, self.output_dim)
        return x

        
class VisConvPatcher(WrapperLikeModuleMixin, AbstractPatcher):
    def __init__(self, 
                patch_size: int,
                input_shape: Tuple[int, int, int],
                num_conv_layers: int = 1,
                ):
        # initialize the parent classes
        WrapperLikeModuleMixin.__init__(self, "conv_patcher")
        AbstractPatcher.__init__(self, patch_size, input_shape)

        # The first convolution performs the patching with a stride equal to the patch size.
        # Subsequent 1x1 convolutions can refine the patch embeddings.
        self.conv_patcher = BasicConvBlock(
            num_conv_layers=num_conv_layers,
            # BasicConvBlock expects a list of channels for each layer.
            # The first conv maps C -> D, subsequent ones are D -> D.
            channels=[self.input_shape[0]] + [self.output_dim for _ in range(num_conv_layers)],
            kernel_sizes=[self.patch_size] + [1 for _ in range(num_conv_layers - 1)],
            strides=[self.patch_size] + [1 for _ in range(num_conv_layers - 1)],
            paddings=0,  # No padding
            use_bn=False,
            activation_after_each_layer=False,
            final_bn_layer=False,
            final_activation=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the convolutional patcher. Output shape: (B, D, n_h, n_w)
        patched_x = self.conv_patcher(x)
        
        # Flatten the spatial dimensions: (B, D, N) where N = n_h * n_w
        x_flattened = patched_x.flatten(2)
        
        # Permute to get (B, N, D) for the attention mechanism
        x_permuted = x_flattened.permute(0, 2, 1)
        
        return x_permuted


class VisUnfoldPatcher(AbstractPatcher):
    """
    Patches an image using the highly efficient `torch.nn.Unfold` operation.
    This is often the most direct and performant way to extract non-overlapping patches.
    """
    def __init__(self, 
                patch_size: int,
                input_shape: Tuple[int, int, int]
                ):        
        super().__init__(patch_size, input_shape)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unfold extracts patches and places them in the last dimension.
        # Input: (B, C, H, W)
        # Output: (B, C * P * P, N) -> (B, D, N)
        unfolded_x = self.unfold(x)
        
        # Permute to get the desired (B, N, D) shape for attention modules.
        # (B, D, N) -> (B, N, D)
        return unfolded_x.permute(0, 2, 1)
    
