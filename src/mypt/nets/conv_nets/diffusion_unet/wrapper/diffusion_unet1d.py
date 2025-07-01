"""
A wrapper to around the UNet1D class to train a diffusion model on 1D conditions.
"""

import torch

from typing import Optional


from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet1d import UNet1DCond
from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding, GaussianFourierEncoding


class ConditionOneDimProcessor(NonSequentialModuleMixin):
    """
    This is a simple conditions processor module that processes the conditions for the diffusion unet.
    """
    def __init__(self,
                 cond_dimension: int,
                 embedding_encoding: torch.nn.Module,
                 embedding_2d_projection: torch.nn.Module,
                 class_embedding: Optional[torch.nn.Embedding] = None):

        NonSequentialModuleMixin.__init__(self, ["embedding_encoding", "embedding_2d_projection", "class_embedding"])

        self.cond_dimension = cond_dimension 

        self.embedding_encoding = embedding_encoding
        self.embedding_2d_projection = embedding_2d_projection
        self.class_embedding = class_embedding

        # if the class embedding is not provided, we will only process the timesteps
        # if the class embedding is provided, we will process the timesteps and the class labels
        self.process_method = self.process_timesteps if self.class_embedding is None else self.process_time_step_and_class_label
        
    def process_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be 1D, got {timesteps.ndim} dimensions")
        
        # each element in the batch should be assigned a timstep
        # each timestep should be 
        # 1. encoded (converted into a vector) (batch_size) -> (batch_size, some_dimension)
        # 2. projected (batch_size, some_dimension) -> (batch_size, cond_dimension)

        return self.embedding_2d_projection(self.embedding_encoding(timesteps))


    def process_time_step_and_class_label(self, time_step: torch.Tensor, class_label: torch.Tensor) -> torch.Tensor:

        # each element in the batch should be assigned a timstep
        # each timestep should be 
        # 1. encoded (converted into a vector) (batch_size) -> (batch_size, some_dimension)
        # 2. projected (batch_size, some_dimension) -> (batch_size, cond_dimension)

        # the class embeddings passed through an embedding layer: (batch_size) -> (batch_size, cond_dimension)
        # this is a simple implementation that can be generalized 

        return self.embedding_2d_projection(self.embedding_encoding(time_step)) + self.class_embedding(class_label)



    def forward(self, time_step: torch.Tensor, *args) -> torch.Tensor:
        # the *args can be either empty or class labels
        return self.process_method(time_step, *args)

    def __call__(self, time_step: torch.Tensor, *args) -> torch.Tensor:
        return self.forward(time_step, *args)
        




class DiffusionUNetOneDim(NonSequentialModuleMixin, torch.nn.Module):
    """
    A wrapper to around the UNet1D class to train a diffusion model on 1D conditions.
    """
    def _set_encoding(self, 
                      embedding_encoding_method: str, 
                      cond_dimension: int):
        
        if embedding_encoding_method not in ['positional', 'gaussian_fourier']:
            raise NotImplementedError(f"embedding_encoding_method {embedding_encoding_method} is not implemented. The only supported methods are 'positional' and 'gaussian_fourier'")
        
        # the initial encoding will map the scalar embedding to cond_dimension * 2
        # then we will use an MLP to map the embedding to the cond_dimension        
        if embedding_encoding_method == 'gaussian_fourier':
            # keep in mind that the GuassianFourierEncoding produces a tensor of shape (batch_size, 2 * embedding_dim) (so embedding_dim must be set to 2 * cond_dimension)
            embedding_encoding = GaussianFourierEncoding(embedding_dim=cond_dimension)
        elif embedding_encoding_method == 'positional':
            # keep in mind that the PositionalEncoding produces a tensor of shape (batch_size, embedding_dim) (so embedding_dim must be set to cond_dimension)
            embedding_encoding = PositionalEncoding(dim_embedding=2 * cond_dimension) 


        # the GenericFCBlock is implemented such that it does not use an activation function at the final layer
        embedding_2d_projection = GenericFCBlock(output=cond_dimension, 
                                                   in_features=2 * cond_dimension, 
                                                   num_layers=2, 
                                                   units=[2 * cond_dimension, 2 * cond_dimension, cond_dimension],
                                                   activation='silu',
                                                   )

        return embedding_encoding, embedding_2d_projection

    def _set_conditions_processor(self,
                                  num_classes: Optional[int] = None,
                                  ):

        if num_classes is not None:
            class_embedding = torch.nn.Embedding(num_classes, self.cond_dimension)
        else:
            class_embedding = None

        embedding_encoding, embedding_2d_projection = self._set_encoding(self.embedding_encoding_method, self.cond_dimension)

        self.conditions_processor = ConditionOneDimProcessor(
            cond_dimension=self.cond_dimension,
            embedding_encoding=embedding_encoding,
            embedding_2d_projection=embedding_2d_projection,
            class_embedding=class_embedding
        )


    def _set_conv_out(self) -> torch.nn.Sequential:
        # this code is inspired by the diffusers library.
        num_groups_out = min(self.unet_output_channels // 4, 32)
        return torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=self.unet_output_channels, num_groups=num_groups_out, eps=1e-5, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv1d(self.unet_output_channels, self.output_channels, kernel_size=3, padding=1)
        )



    def __init__(self, 
                 input_channels: int,
                 output_channels: int,
                 cond_dimension: int, 
                 embedding_encoding_method: str = 'positional',
                 num_classes: Optional[int] = None,
                 *args,
                 **kwargs):
        torch.nn.Module.__init__(self, *args, **kwargs)

        NonSequentialModuleMixin.__init__(self, ["unet", "embedding_encoding", "conditions_processor", "conv_in", "conv_out"])

        self.input_channels = input_channels
        self.unet_output_channels = input_channels * 4 # the number of channels at the output of the inner unet
        self.output_channels = output_channels # the number of channels at the output of the model (diffusion unet)
        self.cond_dimension = cond_dimension
        self.embedding_encoding_method = embedding_encoding_method
        self.num_classes = num_classes
        

        self.unet = UNet1DCond(
            in_channels=input_channels,
            out_channels=self.output_channels,
            cond_dimension=cond_dimension,
        )

        self.conv_in = torch.nn.Conv1d(self.input_channels, self.input_channels, kernel_size=3, padding=1)
        self.conv_out = self._set_conv_out()        
        self._set_conditions_processor(num_classes=num_classes)



