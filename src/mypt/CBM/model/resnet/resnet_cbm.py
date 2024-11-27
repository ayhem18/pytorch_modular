"""
this script contains the implementation of a Concept Bottleneck model on top of a Resnet Backbone.
"""

import os, torch
from typing import Tuple, Union, Iterator, Optional
from torch import nn

from ....backbones.resnetFeatureExtractor import ResNetFeatureExtractor
from ....dimensions_analysis.dimension_analyser import DimensionsAnalyser
from ....linearBlocks import classification_head as ch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class ResnetCBM(nn.Module):
    def set_feature_extractor(self,
                              feature_extractor: ResNetFeatureExtractor) -> nn.Module:
        if not isinstance(feature_extractor, ResNetFeatureExtractor):
            raise ValueError(f"The ResnetCBM class expects a feature extractor of type {ResNetFeatureExtractor}. Found: {type(feature_extractor)}")

        try:
            # make sure the feature extractor accepts tensors with the same input shape
            x_temp = torch.rand(*((1,) + self.input_shape))
            feature_extractor.forward(x_temp)
        
        except ValueError:
            raise ValueError(f"The feature extractor passed expects a different input shape than the one passed to the CBM. ")

        return feature_extractor

    def set_concept_projection(self,
                               concept_projection: nn.Module = None,
                               num_concept_layers: int = 2,
                               dropout: float = None):

        # this function assumes that set_feature_extractor has already been called
        if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
            raise ValueError(f"The field 'feature_extractor' must be set before calling this function")

        # extract the number of input units to the concepts projection layer
        da = DimensionsAnalyser()
        # the analyse_dimensions method will return (10, in_features) (10 is the batch size is this case)
        _, in_features = da.analyse_dimensions(input_shape=((10,) + self.input_shape),
                                               net=nn.Sequential(self.feature_extractor, nn.Flatten()),
                                               method='static')

        cp = concept_projection if concept_projection is not None \
            else ch.ExponentialClassifier(num_classes=self.num_concepts,
                                          in_features=in_features,
                                          num_layers=num_concept_layers, 
                                          dropout=dropout)

        # let's make sure the linear block accepts tensors of the expected shape
        try:
            x_temp = torch.randn(10, in_features)
            cp.forward(x_temp)
        except ValueError:
            raise ValueError(f"The concept projection block expects tensors "
                             f"of shape different from {(None, in_features)}")

        return cp

    def set_classification_head(self,
                                classifier: nn.Module,
                                num_layers: int = 2):
        # this function assumes that
        # both set_feature_extractor and set_concept_projection methods have been called
        if (not hasattr(self, 'feature_extractor') or self.feature_extractor is None or
                not hasattr(self, 'concept_projection') or self.concept_projection is None):
            raise ValueError(f"The fields 'feature_extractor' and 'concept_projection'"
                             f"must be set before calling this function")

        classifier = classifier if classifier is not None else ch.ExponentialClassifier(
            num_classes=self.num_classes, in_features=self.num_concepts, num_layers=num_layers)

        # make sure the classification head is compatible with the rest of the model
        try:
            x_temp = torch.randn(10, self.num_concepts)
            classifier.forward(x_temp)
        except ValueError:
            raise ValueError(f"The classification head expects tensors of shape different from "
                             f"{(None, self.num_concepts)}")
        return classifier

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_concepts: int,
                 num_classes: int,

                 feature_extractor: Union[str, nn.Module] = 'resnet',
                 concept_projection: nn.Module = None,
                 classification_head: nn.Module = None,

                 num_concept_layers: int = 2,
                 num_classification_layers: int = 2,
                 num_fe_blocks: int = -1,
                 blocks_to_freeze: Union[bool, int, float] = True,
                 dropout:Optional[float] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        # verify the input
        if not (isinstance(input_shape, Tuple) and len(input_shape) in [2, 3]):
            raise ValueError(f"the input shape is expected to be either of {2} or {3} dimensions\n"
                             f"Found {input_shape}")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.output_units = self.num_classes if self.num_classes > 2 else 1
        self.num_concepts = num_concepts

        if isinstance(feature_extractor, str):
            # the default will be to extract the entire thing
            feature_extractor = ResNetFeatureExtractor(num_layers=num_fe_blocks,
                                                       freeze=blocks_to_freeze)
            
        # set the feature extractor
        self.feature_extractor = self.set_feature_extractor(feature_extractor)

        # set the concept_projection layer
        self.concept_projection = self.set_concept_projection(concept_projection=concept_projection,
                                                              num_concept_layers=num_concept_layers, 
                                                              dropout=dropout)
        # set the classification head
        self.classification_head = self.set_classification_head(classifier=classification_head,
                                                                num_layers=num_classification_layers)
        self.flatten_layer = nn.Flatten()

        # set the model
        self.model = nn.Sequential(self.feature_extractor,
                                   self.flatten_layer,
                                   self.concept_projection,
                                   self.classification_head)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x_temp = x[0] if len(x.shape) > 3 else x
        if x_temp.shape != self.input_shape:
            raise ValueError(f"The model expects input of the shape {self.input_shape}")

        concept_logits = self.concept_projection.forward(self.feature_extractor.forward(x).squeeze())
        class_logits = self.classification_head(concept_logits) 
        
        # make sure to return both types of logits
        return concept_logits, class_logits

    def children(self) -> Iterator[nn.Module]:
        return self.model.children()

    def named_children(self) -> Iterator[tuple[str, nn.Module]]:
        return self.model.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()

    def __str__(self) -> str:
        # the print statement will be much clearer after setting the __str__ method
        return self.model.__str__()

    def __repr__(self) -> str:
        # the print statement will be much clearer after setting the __str__ method
        return self.model.__repr__()

    