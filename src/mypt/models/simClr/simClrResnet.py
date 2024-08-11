"""
This script contains Implementation 
"""

import torch

from typing import Union, Tuple, Optional, Iterator
from torch import nn


from mypt.backbones import resnetFeatureExtractor as rfe
from mypt.linearBlocks import classification_head as ch
from mypt.dimensions_analysis import dimension_analyser as da

from .simClrModel import SimClrModel


class ResnetSimClr(SimClrModel):
    def __init__(self, 
                    input_shape: Tuple[int, int, int],
                    output_dim: int,
                    num_fc_layers: int,
                    dropout: Optional[float] = None,
                    fe_num_blocks: int=-1, # use all the layer blocks of the Resnet feature extractor
                    architecture: int = 50, # use Resnet50
                    freeze: Union[int, bool]=False, # do not freeze any of the  layers of the pretrained model, 
                    freeze_layers: bool=True) -> None:

        super().__init__()

        # the feature extractor or the encoder "f" as denoted in the paper.
        self.fe = rfe.ResNetFeatureExtractor(num_layers=fe_num_blocks, 
                                        architecture=architecture,
                                        freeze=freeze, 
                                        freeze_layers=freeze_layers, 
                                        add_fc=False,)

        dim_analyser = da.DimensionsAnalyser(method='static')

        # calculate the number of features passed to the classification head.
        _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=nn.Sequential(self.fe, nn.Flatten()))

        # calculate the output of the
        self.ph = ch.ExponentialClassifier(num_classes=output_dim, 
                                           in_features=in_features, 
                                           num_layers=num_fc_layers, 
                                           dropout=dropout)

        self.flatten_layer = nn.Flatten()
