"""
This script contains the implementation of a CBM model composed of a feature extractor, a concept projection head and a classification head.

The implementation extends the original implementation suggested in the paper: 

https://proceedings.mlr.press/v119/koh20a.html 


where the model is not restricted to a single sparse layer (for interpretability purposes). The CBM can have as many layers in the fully connected part 
to ensure sufficient capacity for both tasks (concepts and labels predictions.)
"""

import torch

from torch import nn
from abc import abstractmethod
from typing import Optional, Tuple


from ...dimensions_analysis.dimension_analyser import DimensionsAnalyser
from ...building_blocks.linear_blocks.fc_blocks import ExponentialFCBlock
from ...building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin

@abstractmethod
class AbstractCBM(WrapperLikeModuleMixin):

    @abstractmethod
    def _set_feature_extractor(self,
                              feature_extractor: nn.Module) -> nn.Module:
        try:
            # make sure the feature extractor accepts tensors with the same input shape
            x_temp = torch.rand(*((1,) + self.input_shape))
            feature_extractor.forward(x_temp)
        
        except ValueError:
            raise ValueError(f"The feature extractor passed expects a different input shape than the one passed to the CBM. ")

        return feature_extractor


    # leaving the _set_concept_projection method as abstract to enforce concious use of the parent implementation
    # in other words, if the child class uses the parent implementation, it should be a trival implementation that calls super()._set_concept_projection() 
    @abstractmethod
    def _set_concept_projection(self,
                               concept_projection: Optional[nn.Module] = None,
                               num_concept_layers: int = 2,
                               dropout: Optional[float] = None):


        # this function assumes that set_feature_extractor has already been called
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            raise ValueError(f"The field 'feature_extractor' must be set before calling this function")


        if concept_projection is not None:
            return concept_projection 
        

        # extract the number of input units to the concepts projection layer
        da = DimensionsAnalyser()
        # the analyse_dimensions method will return (10, in_features) (10 is the batch size is this case)
        _, in_features = da.analyse_dimensions(input_shape=((10,) + self._input_shape),
                                               net=nn.Sequential(self._feature_extractor, nn.Flatten()),
                                               method='static')

        cp = ExponentialFCBlock(output=self._num_concepts,
                                in_features=in_features,
                                num_layers=num_concept_layers, 
                                dropout=dropout)

        return cp


    @abstractmethod
    def _set_classification_head(self,
                                classifier: Optional[nn.Module] = None,
                                num_layers: int = 2,
                                dropout: Optional[float] = None):

        # this function assumes that
        # both set_feature_extractor and set_concept_projection methods have been called
        if (not hasattr(self, 'feature_extractor') or self.feature_extractor is None or
                not hasattr(self, 'concept_projection') or self.concept_projection is None):
            raise ValueError(f"The fields 'feature_extractor' and 'concept_projection'"
                             f"must be set before calling this function")

        if classifier is not None:
            return classifier

        classifier = ExponentialFCBlock(
            output=self._output_units, 
            in_features=self._num_concepts, 
            num_layers=num_layers,
            dropout=dropout)

        return classifier




    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_concepts: int,
                 num_classes: int,

                 # in case of passing concrete modules
                 feature_extractor: nn.Module,
                 concept_projection: Optional[nn.Module] = None,
                 classification_head: Optional[nn.Module] = None,
                 
                 # those parameters can be passed to the inner setter methods to create the modules
                 num_concept_layers: int = 2, 
                 num_classification_layers: int = 2,
                 dropout: Optional[float] = None,

                 *args, **kwargs):
    
        # the initialization is common

        super().__init__('_model', *args, **kwargs)
        
        self._input_shape = input_shape
        self._num_concepts = num_concepts
        self._num_classes = num_classes
        self._output_units = self._num_classes if self._num_classes > 2 else 1
        
    
        # set the feature extractor
        self._feature_extractor = self._set_feature_extractor(feature_extractor)

        # set the concept_projection layer
        self._concept_projection = self._set_concept_projection(concept_projection=concept_projection,
                                                              num_concept_layers=num_concept_layers, 
                                                              dropout=dropout)
        # set the classification head
        self._classification_head = self._set_classification_head(classifier=classification_head,
                                                                  num_layers=num_classification_layers,
                                                                  dropout=dropout)
        self._flatten_layer = nn.Flatten()

        # set the model
        self._model = nn.Sequential(self._feature_extractor,
                                   self._flatten_layer,
                                   self._concept_projection,
                                   self._classification_head)

        

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concept_logits = self._concept_projection.forward(self._feature_extractor.forward(x).squeeze())
        class_logits = self._classification_head(concept_logits) 
        
        # make sure to return both types of logits
        return concept_logits, class_logits


    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape
    
    @property
    def num_concepts(self) -> int:
        return self._num_concepts
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    


    # @property
    # def feature_extractor(self) -> nn.Module:
    #     return self._feature_extractor
    
    # @property
    # def concept_projection(self) -> nn.Module:
    #     return self._concept_projection
    
    # @property
    # def classification_head(self) -> nn.Module:
    #     return self._classification_head

    # @property
    # def flatten_layer(self) -> nn.Module:
    #     return self._flatten_layer


