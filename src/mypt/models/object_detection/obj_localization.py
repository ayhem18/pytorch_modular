"""
This script contains the implementation of certain models based on different backbones
"""

import torch
from typing import Tuple, Iterator, Optional

from mypt.backbones import alexnetFeatureExtractor as afe
from mypt.backbones import resnetFeatureExtractor as rfe
from mypt.linearBlocks import classification_head as ch
from mypt.dimensions_analysis import dimension_analyser as da


class ObjectLocalizationModel(torch.torch.nn.Module):
    def __init__(self, num_classes:int) -> None:
        super().__init__()

        self.num_classes = num_classes
        # the feature extractor or the encoder "f" as denoted in the paper.
        self.fe: torch.nn.Module = None 
        self.flatten_layer = torch.nn.Flatten()
        self.head: torch.nn.Module = None
        self.model = torch.nn.Sequential(self.fe, self.flatten_layer, self.head)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x)


    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__() 
    
    def children(self) -> Iterator[torch.nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.model
        return self.model.children()

    def modules(self) -> Iterator[torch.nn.Module]:
        return self.model.modules()
    
    def named_children(self) -> Iterator[Tuple[str, torch.nn.Module]]:
        return self.model.named_children()

    def to(self, *args, **kwargs):
        self.fe = self.fe.to(*args, **kwargs)
        self.flatten_layer = self.flatten_layer.to(*args, **kwargs)
        self.head = self.head.to(*args, **kwargs)
        return self 

    def __call__(self, x: torch.Tensor):
        return self.forward(x)


class AlexnetObjectLocalization(ObjectLocalizationModel):
    def __init__(self,                 
                input_shape: Tuple[int, int, int],
                num_classes:int,
                num_fc_layers: int,
                dropout: Optional[float] = None,
                freeze: bool=False, # do not freeze any of the  layers of the pretrained model, 
                ) -> None:

        super().__init__(num_classes=num_classes)

        self.fe = afe.AlexNetFeatureExtractor(model_blocks='conv_block_adapool',
                                              frozen_model_blocks=freeze)

        dim_analyser = da.DimensionsAnalyser(method='static')

        # calculate the number of features passed to the classification head.
        _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=torch.nn.Sequential(self.fe, torch.nn.Flatten()))

        # calculate the output of the
        self.head = ch.ExponentialClassifier(num_classes=self.num_classes + 5, # 5 represent 4 units for bounding box + 1 unit as an object indicator
                                           in_features=in_features, 
                                           num_layers=num_fc_layers, 
                                           dropout=dropout)

        self.model = torch.nn.Sequential(self.fe, self.flatten_layer, self.head)
