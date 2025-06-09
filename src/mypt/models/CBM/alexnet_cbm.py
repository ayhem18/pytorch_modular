"""
this script contains the implementation of a Concept Bottleneck model on top of a Resnet Backbone.
"""

from torch import nn
from typing import List, Tuple, Union, Optional


from .abstractCBM import AbstractCBM
from ...backbones.alexnetFE import AlexNetFE


class AlexnetCBM(AbstractCBM):
    def _set_feature_extractor(self,
                              feature_extractor: nn.Module) -> nn.Module:
        if not isinstance(feature_extractor, AlexNetFE):
            raise ValueError(f"The AlexnetCBM class expects a feature extractor of type {AlexNetFE}. Found: {type(feature_extractor)}")

        return super()._set_feature_extractor(feature_extractor=feature_extractor)


    def _set_concept_projection(self,
                               concept_projection: Optional[nn.Module] = None,
                               num_concept_layers: int = 2,
                               dropout: Optional[float] = None):

        return super()._set_concept_projection(concept_projection=concept_projection,
                                               num_concept_layers=num_concept_layers,
                                               dropout=dropout)

    def _set_classification_head(self,
                                classifier: Optional[nn.Module] = None,
                                num_layers: int = 2,
                                dropout: Optional[float] = None):
        return super()._set_classification_head(classifier=classifier,
                                                num_layers=num_layers,
                                                dropout=dropout)



    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_concepts: int,
                 num_classes: int,

                 # arguments to initialize the alexnet Feature Extractor
                 alexnet_fe_blocks: Union[str, List[str], int, List[int]],
                 alexnet_fe_frozen_blocks: Union[str, List[str], int, List[int]],

                 concept_projection: Optional[nn.Module] = None,
                 classification_head: Optional[nn.Module] = None,

                 num_concept_layers: int = 2,
                 num_classification_layers: int = 2,
                 dropout: Optional[float] = None):

        # before calling the super().__init__ method, it might be necessary to initialize a feature Extractor 
        feature_extractor = AlexNetFE(model_blocks=alexnet_fe_blocks,
                                      frozen_model_blocks=alexnet_fe_frozen_blocks)

        super().__init__(
                         input_shape=input_shape,
                         num_concepts=num_concepts,
                         num_classes=num_classes,
                         feature_extractor=feature_extractor,
                         concept_projection=concept_projection,
                         classification_head=classification_head,
                         num_concept_layers=num_concept_layers,
                         num_classification_layers=num_classification_layers,
                         dropout=dropout)
        






    