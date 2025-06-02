from mypt.building_blocks.conv_blocks.basic.conv_block import BasicConvBlock
from mypt.building_blocks.conv_blocks.residual.resnet_block import WideResnetBlock
import torch

from torch import nn
from typing import List, Optional, Tuple, Union, Callable


from mypt.building_blocks.linear_blocks.fc_blocks import GenericFCBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet1d import UNet1DCond
from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet3d import UNet3DCond
from mypt.nets.conv_nets.diffusion_unet.unet.abstract_unet import AbstractUNetCond
from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding, GaussianFourierEncoding


class EmbeddingProjection(nn.Module):
    """
    This is a simple embedding projection module that projects a scalar embedding to a higher dimension.

    Args:
        in_embed_dim (int): The dimension of the input embedding.
        out_embed_dim (int): The dimension of the output embedding.
    """
    def __init__(self, 
                 in_embed_dim: int, 
                 out_embed_dim: int,
                *args,
                **kwargs):

        super().__init__(*args, **kwargs)
        self.embedding_projection = nn.Linear(in_embed_dim, out_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(x)


class ConditionsProcessor(nn.Module):
    """
    This is a simple conditions processor module that processes the conditions for the diffusion unet.
    """
    def __init__(self,
                 cond_dimension: int,
                 embedding_encoding: nn.Module,
                 embedding_2d_projection: nn.Module,
                 embedding_3d_projection: Optional[nn.Module] = None,
                 class_embedding: Optional[nn.Embedding] = None,
                 condition_3d_shape: Optional[Tuple[int, ...]] = None,
                 condition_3d_label_map: bool = True,
                 *args,
                 **kwargs):


        if (embedding_3d_projection is None) != (condition_3d_shape is None):
            raise ValueError("if embedding_3d_projection is provided, condition_3d_shape must also be provided and vice versa")


        if (condition_3d_shape is not None and condition_3d_label_map) and class_embedding is None:  
            raise ValueError("if the model is conditioned on a label map, class_embedding must be provided") 


        super().__init__(*args, **kwargs)

        self.cond_dimension = cond_dimension 

        self.embedding_encoding = embedding_encoding

        self.embedding_2d_projection = embedding_2d_projection
        self.embedding_3d_projection = embedding_3d_projection

        self.class_embedding = class_embedding 

        self.condition_3d_shape = condition_3d_shape
        self.condition_3d_label_map = condition_3d_label_map


        if class_embedding is None and condition_3d_shape is None: 
            self.process_method = self.process_only_time_step
        elif class_embedding is not None and condition_3d_shape is None: 
            self.process_method = self.process_time_step_and_class_label
        elif class_embedding is None and condition_3d_shape is not None: 
            self.process_method = self.process_time_step_and_3d_condition
        else:
            self.process_method = self.process_all


    
    def process_only_time_step(self, time_step: torch.Tensor, *args) -> torch.Tensor:
        # pass the time step through the embedding encoding and the embedding projection 
        return self.embedding_2d_projection(self.embedding_encoding(time_step))

    def process_time_step_and_class_label(self, time_step: torch.Tensor, class_label: torch.Tensor) -> torch.Tensor:
        # pass the time step and the class label through the embedding encoding and the embedding projection 
        # pass each of them through the embedding encoding, sum them up and then pass the result through the embedding projection 
        return self.embedding_2d_projection(self.embedding_encoding(time_step)) + self.class_embedding(class_label)

    def _process_label_map(self, label_map: torch.Tensor) -> torch.Tensor:

        if label_map.ndim == 3: 
            bs, h, w = label_map.shape

        elif label_map.ndim == 4:
            bs, c, h, w = label_map.shape
            if c != 1: 
                raise ValueError(f"if the label map is 4D, it must have 1 channel, got {c} channels")

        else:
            raise ValueError(f"label_map must be 3D or 4D, got {label_map.ndim} dimensions")

        label_map_emb = self.class_embedding.forward(label_map.reshape(label_map.shape[0], -1)).reshape(bs, self.cond_dimension, h, w)

        return label_map_emb    

    def process_time_step_and_3d_condition(self, time_step: torch.Tensor, cond_3d: torch.Tensor) -> torch.Tensor:
        time_step_emb = self.embedding_2d_projection(self.embedding_encoding(time_step))

        if self.condition_3d_label_map: 
            cond_3d_emb = self._process_label_map(cond_3d)
        else:
            cond_3d_emb = self.embedding_3d_projection(cond_3d)

        time_step_emb = time_step_emb[:, :, None, None].repeat(1, 1, cond_3d_emb.shape[2], cond_3d_emb.shape[3])

        return time_step_emb + cond_3d_emb

    def process_all(self, time_step: torch.Tensor, class_label: torch.Tensor, cond_3d: torch.Tensor) -> torch.Tensor:
        cond_2d_emb = self.process_time_step_and_class_label(time_step, class_label)

        if self.condition_3d_label_map: 
            cond_3d_emb = self._process_label_map(cond_3d)
        else:
            cond_3d_emb = self.embedding_3d_projection(cond_3d)

        cond_2d_emb = cond_2d_emb[:, :, None, None].repeat(1, 1, cond_3d_emb.shape[2], cond_3d_emb.shape[3])

        return cond_2d_emb + cond_3d_emb


    def forward(self, time_step: torch.Tensor, *args) -> torch.Tensor:
        return self.process_method(time_step, *args)

    def __call__(self, time_step: torch.Tensor, *args) -> torch.Tensor:
        return self.forward(time_step, *args)



class DiffusionUNet(NonSequentialModuleMixin, nn.Module):
    def _set_encoding(self, embedding_encoding_method: str, 
                      cond_dimension: int, 
                      condition_3d_shape: Optional[Tuple[int, ...]] = None):
        
        if embedding_encoding_method not in ['positional', 'gaussian_fourier']:
            raise NotImplementedError(f"embedding_encoding_method {embedding_encoding_method} is not implemented. The only supported methods are 'positional' and 'gaussian_fourier'")
        
        # the initial encoding will map the scalar embedding to cond_dimension * 2
        # then we will use an MLP to map the embedding to the cond_dimension        
        if embedding_encoding_method == 'gaussian_fourier':
            # keep in mind that the GuassianFourierEncoding produces a tensor of shape (batch_size, 2 * embedding_dim) (so embedding_dim must be set to 2 * cond_dimension)
            self.embedding_encoding = GaussianFourierEncoding(embedding_dim=cond_dimension)
        elif embedding_encoding_method == 'positional':
            # keep in mind that the PositionalEncoding produces a tensor of shape (batch_size, embedding_dim) (so embedding_dim must be set to cond_dimension)
            self.embedding_encoding = PositionalEncoding(dim_embedding=2 * cond_dimension) 


        # the GenericFCBlock is implemented such that it does not use an activation function at the final layer
        self.embedding_2d_projection = GenericFCBlock(output=cond_dimension, 
                                                   in_features=2 * cond_dimension, 
                                                   num_layers=2, 
                                                   units=[2 * cond_dimension, 2 * cond_dimension, cond_dimension],
                                                   activation='relu',
                                                   )

        if condition_3d_shape is not None:
            self.embedding_3d_projection = WideResnetBlock(
                in_channels=condition_3d_shape[0],
                out_channels=cond_dimension,
                stride=1,
                dropout_rate=0.0, # no need for a dropout layer
            )
        else:
            self.embedding_3d_projection = None
        

    def _set_conditions_processor(self,
                                  num_classes: Optional[int] = None,
                                  condition_3d_shape: Optional[Tuple[int, ...]] = None, 
                                  condition_3d_label_map: bool = True):

        # if the unet will be conditioned on a label map, the number of classes must be passed
        if (condition_3d_shape is not None and condition_3d_label_map) and num_classes is None:
            raise ValueError("if the model is conditioned on a label map, the number of classes must be provided")

        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, self.cond_dimension)

        self.conditions_processor = ConditionsProcessor(
            cond_dimension=self.cond_dimension,
            embedding_encoding=self.embedding_encoding,
            embedding_2d_projection=self.embedding_2d_projection,
            embedding_3d_projection=self.embedding_3d_projection,
            class_embedding=self.class_embedding,
            condition_3d_shape=condition_3d_shape,
            condition_3d_label_map=condition_3d_label_map,
        )



    def __init__(self,
                input_channels: int,
                output_channels: int,
                cond_dimension: int, 
                embedding_encoding_method: str = 'positional',
                num_classes: Optional[int] = None,
                condition_3d_shape: Optional[Tuple[int, ...]] = None,
                condition_3d_label_map: bool = True,
                *args,
                **kwargs):
        
        nn.Module.__init__(self, *args, **kwargs)

        # no need to set the embedding_2d_projection and embedding_3d_projection as they are part of the conditions_processor.
        NonSequentialModuleMixin.__init__(self, ["unet", "embedding_encoding", "conditions_processor"])

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cond_dimension = cond_dimension

        self.embedding_encoding: nn.Module = None 
        self.embedding_2d_projection: GenericFCBlock = None 
        
        self.class_embedding: nn.Embedding = None
        self.embedding_3d_projection: WideResnetBlock = None 
    

        self.conditions_processor: ConditionsProcessor = None 
        self.unet: AbstractUNetCond = None 


        self._set_encoding(embedding_encoding_method, cond_dimension, condition_3d_shape)
        self._set_conditions_processor(num_classes=num_classes, condition_3d_shape=condition_3d_shape, condition_3d_label_map=condition_3d_label_map)


        # if label_map_shape is not None, the model will be conditioned on a label map
        if condition_3d_shape is not None:
            if condition_3d_label_map and condition_3d_shape[0] != 1:
                    raise ValueError(f"if the model is conditional on a label map, the first dimension of the label map shape must be 1, got {condition_3d_shape[0]}")

            self.unet = UNet3DCond(
                in_channels=input_channels,
                out_channels=output_channels,
                cond_dimension=cond_dimension,
            )

        else:
            self.unet = UNet1DCond(
                in_channels=input_channels,
                out_channels=output_channels,
                cond_dimension=cond_dimension,
            )


    def build_down_block(
        self,
        num_down_layers: int,
        num_res_blocks: int,
        out_channels: List[int],
        downsample_types: Union[str, List[str]] = "conv",
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """Build the downsampling block in the UNet architecture."""
        
        # Call the underlying UNet implementation
        self.unet.build_down_block(
            num_down_layers=num_down_layers,
            num_resnet_blocks=num_res_blocks,
            out_channels=out_channels,
            downsample_types=downsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual,
        )

    def build_middle_block(
        self,
        num_res_blocks: int,
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """Build the middle block in the UNet architecture."""
        self.unet.build_middle_block(
            num_resnet_blocks=num_res_blocks,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual,
        )

    def build_up_block(
        self,
        num_res_blocks: int,
        upsample_types: Union[str, List[str]] = "transpose_conv",
        inner_dim: int = 256,
        dropout_rate: float = 0.0,
        norm1: Optional[nn.Module] = None,
        norm1_params: Optional[dict] = None,
        norm2: Optional[nn.Module] = None,
        norm2_params: Optional[dict] = None,
        activation: Optional[Union[str, Callable]] = None,
        activation_params: Optional[dict] = None,
        film_activation: Union[str, Callable] = "relu",
        film_activation_params: dict = {},
        force_residual: bool = False,
    ):
        """Build the upsampling block in the UNet architecture."""
        
        self.unet.build_up_block(
            num_resnet_blocks=num_res_blocks,
            upsample_types=upsample_types,
            inner_dim=inner_dim,
            dropout_rate=dropout_rate,
            norm1=norm1,
            norm1_params=norm1_params,
            norm2=norm2,
            norm2_params=norm2_params,
            activation=activation,
            activation_params=activation_params,
            film_activation=film_activation,
            film_activation_params=film_activation_params,
            force_residual=force_residual,
        )

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, *args) -> torch.Tensor:
        return self.unet.forward(x, self.conditions_processor(time_step, *args))

    def __call__(self, x: torch.Tensor, time_step: torch.Tensor, *args) -> torch.Tensor:
        return self.forward(x, time_step, *args)