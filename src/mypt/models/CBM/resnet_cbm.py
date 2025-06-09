"""
this script contains the implementation of a Concept Bottleneck model on top of a Resnet Backbone.
"""

from torch import nn
from typing import Tuple, Union, Optional


from .abstractCBM import AbstractCBM
from ...backbones.resnetFE import ResnetFE


class ResnetCBM(AbstractCBM):
    def _set_feature_extractor(self,
                              feature_extractor: nn.Module) -> nn.Module:
        if not isinstance(feature_extractor, ResnetFE):
            raise ValueError(f"The ResnetCBM class expects a feature extractor of type {ResnetFE}. Found: {type(feature_extractor)}")

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

                 # arguments to initialize the resnet Feature Extractor
                 build_by_layer: bool,
                 num_extracted_layers: int,
                 num_extracted_bottlenecks,
                 freeze: Union[bool, int],
                 freeze_by_layer: bool,

                 add_global_average: bool = True,
                 architecture: int = 50,

                 concept_projection: Optional[nn.Module] = None,
                 classification_head: Optional[nn.Module] = None,

                 num_concept_layers: int = 2,
                 num_classification_layers: int = 2,
                 concept_projection_dropout: Optional[float] = None,


                 *args, **kwargs):

        # before calling the super().__init__ method, it might be necessary to initialize a feature Extractor 
        feature_extractor = ResnetFE(build_by_layer=build_by_layer,
                                        num_extracted_layers=num_extracted_layers,
                                        num_extracted_bottlenecks=num_extracted_bottlenecks,
                                        freeze=freeze,
                                        freeze_by_layer=freeze_by_layer,
                                        add_global_average=add_global_average,
                                        architecture=architecture)

        super().__init__(
                         input_shape=input_shape,
                         num_concepts=num_concepts,
                         num_classes=num_classes,
                         feature_extractor=feature_extractor,
                         concept_projection=concept_projection,
                         classification_head=classification_head,
                         num_concept_layers=num_concept_layers,
                         concept_projection_dropout=concept_projection_dropout,
                         num_classification_layers=num_classification_layers,

                         *args, **kwargs)
        






    