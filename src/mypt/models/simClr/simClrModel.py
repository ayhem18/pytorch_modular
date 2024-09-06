"""
This script contains an abstact class of a model trained with the SimClr algorithm.
"""

import torch

from torch import nn
from typing import Union, Tuple, Optional, Iterator

from ...backbones import alexnetFeatureExtractor as afe
from ...backbones import resnetFeatureExtractor as rfe
from ...linearBlocks import classification_head as ch
from ...dimensions_analysis import dimension_analyser as da


class SimClrModel(nn.Module):
    def __init__(self) -> None:
        
        super().__init__()
        
        # the feature extractor or the encoder "f" as denoted in the paper.
        self.fe: nn.Module = None 
        self.ph: nn.Module = None
        self.flatten_layer = nn.Flatten()
        self.model = nn.Sequential(self.fe, self.flatten_layer, self.ph)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # the forward function returns both f(x_i) and g(f(x_i)), any loss object should work with gf(x_i)
        f_xi = self.flatten_layer.forward(self.fe(x))
        return f_xi, self.ph.forward(f_xi)


    def __str__(self):
        return self.model.__str__()
    
    def __repr__(self):
        return self.model.__repr__() 
    
    def children(self) -> Iterator[nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.model
        return self.model.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.model.named_children()

    def to(self, *args, **kwargs):
        # self.model = self.model.to(*args, **kwargs)
        self.fe = self.fe.to(*args, **kwargs)
        self.flatten_layer = self.flatten_layer.to(*args, **kwargs)
        self.ph = self.ph.to(*args, **kwargs)
        return self 

    def __call__(self, x: torch.Tensor):
        return self.forward(x)


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



class AlexnetSimClr(SimClrModel):
    def __init__(self, 
                    input_shape: Tuple[int, int, int],
                    output_dim: int,
                    num_fc_layers: int,
                    dropout: Optional[float] = None,
                    freeze: bool=False, # do not freeze any of the  layers of the pretrained model, 
                    ) -> None:

        super().__init__()

        self.fe = afe.AlexNetFeatureExtractor(model_blocks='conv_block_adapool',
                                              frozen_model_blocks=freeze)

        dim_analyser = da.DimensionsAnalyser(method='static')

        # calculate the number of features passed to the classification head.
        _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=nn.Sequential(self.fe, nn.Flatten()))

        # calculate the output of the
        self.ph = ch.ExponentialClassifier(num_classes=output_dim, 
                                           in_features=in_features, 
                                           num_layers=num_fc_layers, 
                                           dropout=dropout)
