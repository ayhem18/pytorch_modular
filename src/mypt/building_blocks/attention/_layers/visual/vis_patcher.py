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


class AbstractOverlappingPatcher(torch.nn.Module, ABC):
    """
    Abstract base class for patchers that create overlapping patches.
    This is achieved by using a stride smaller than the patch size.

    This technique increases the number of tokens (sequence length) fed to the
    Transformer, which can improve performance by providing richer local context,
    at the cost of increased computational complexity.
    """
    def __init__(self,
                 patch_size: int,
                 stride: int,
                 input_shape: Tuple[int, int, int]
                 ):
        super().__init__()
        if stride >= patch_size:
            raise ValueError(f"Stride must be less than patch size for overlapping patches. Got stride {stride} and patch size {patch_size}.")
        self.patch_size = patch_size
        self.stride = stride
        self.input_shape = input_shape
        # N: number of patches, calculated with stride
        self.num_patches_h = (input_shape[1] - patch_size) // stride + 1
        self.num_patches_w = (input_shape[2] - patch_size) // stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w
        # D: output dimension
        self.output_dim = input_shape[0] * patch_size * patch_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class AbstractConvStemPatcher(torch.nn.Module, ABC):
    """
    Abstract base class for a hybrid patcher that uses a "convolutional stem".

    Instead of a single patching operation, this approach uses several layers of
    standard convolutions (often with stride 2) to progressively downsample the
    image and extract low-level features. The output of this stem is then
    flattened and treated as the sequence of input tokens for the Transformer.

    This combines the efficiency and inductive bias of CNNs for low-level
    vision with the global context modeling of Transformers.

    References:
        - CoAtNet: Marrying Convolution and Attention for All Data Sizes
          (https://arxiv.org/abs/2106.04803)
        - Early Convolutions Help Transformers See Better
          (https://arxiv.org/abs/2106.14881)
    """
    def __init__(self,
                 stem_config: dict, # Could contain channels, kernels, strides etc.
                 input_shape: Tuple[int, int, int]
                 ):
        super().__init__()
        self.stem_config = stem_config
        self.input_shape = input_shape
        # Output shape, num_patches, and output_dim would be determined
        # by the specific stem architecture.
        # These would need to be calculated in the concrete implementation.
        self.output_shape = None # To be calculated
        self.num_patches = None # To be calculated
        self.output_dim = None # To be calculated

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class AbstractMergingPatcher(torch.nn.Module, ABC):
    """
    Abstract base class for patchers that support hierarchical feature
    representation through patch merging or pooling.

    This strategy is central to architectures like the Swin Transformer.
    It starts with small patches and, after a set of Transformer blocks,
    merges adjacent patches to create fewer, larger patches for the next
    stage. This creates a hierarchical representation that captures features
    at multiple scales, similar to a traditional CNN, while maintaining
    efficient computation.

    The 'merge' or 'pool' operation is what distinguishes this class.

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (https://arxiv.org/abs/2103.14030)
    """
    def __init__(self,
                 initial_patch_size: int,
                 input_shape: Tuple[int, int, int]
                ):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.input_shape = input_shape

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the initial input tensor."""
        pass

    @abstractmethod
    def merge(self, x: torch.Tensor) -> torch.Tensor:
        """Merges existing patches to create a new, downsampled sequence."""
        pass
    
