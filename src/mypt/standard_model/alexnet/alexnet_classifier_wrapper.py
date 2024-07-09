"""
This script contains the implementation of Pytorch Lightning wrapper around a standard classifier with a Resnet backbone.
"""

import torch
from torch import nn
from typing import Union, Tuple, Optional, List, Dict, Any


from ..backbone_classifier_wrapper import BackboneClassifierWrapper
from ...backbones import alexnetFeatureExtractor as afe
from ...linearBlocks import classification_head as ch
from ...dimensions_analysis import dimension_analyser as da
from ...code_utilities import str_class_map as scm 
from ...code_utilities import pytorch_utilities as pu 


class AlexnetClassifierWrapper(BackboneClassifierWrapper):
    def __init__(self, 
                input_shape: Tuple,
                num_classes: int,   
                num_classification_layers: int, 

                feature_extractor_blocks:Union[str, List[str], int, List[int]],
                frozen_feature_extractor_blocks:Union[str, List[str], int, List[int]],
                optimizer_class: callable,

                learning_rate: Optional[float] = 10 ** -4,
                num_vis_images: int = 3,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_class: Optional[callable] = None,
                scheduler_keyargs: Optional[Dict] = None   
                 ):
        
        # parent class constructor
        super().__init__(
                        input_shape=input_shape, 
                        num_classes=num_classes, 

                        optimizer_class=optimizer_class,
                        learning_rate=learning_rate,
                        scheduler_class=scheduler_class,

                        num_vis_images=num_vis_images,
                        scheduler_keyargs=scheduler_keyargs,
                        optimizer_keyargs=optimizer_keyargs                
                        )
        
        self.fe = afe.AlexNetFeatureExtractor(model_blocks=feature_extractor_blocks, 
                                              frozen_model_blocks=frozen_feature_extractor_blocks)
        self.flatten_layer = nn.Flatten()

        # calculate the number of features to pass
        dim_analyser = da.DimensionsAnalyser(method='static')

        # calculate the number of features passed to the classification head.
        _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=nn.Sequential(self.fe, nn.Flatten()))

        # initialize the classification head
        self.classification_head = ch.ExponentialClassifier(num_classes=num_classes, 
                                                            in_features=in_features, 
                                                            num_layers=num_classification_layers)

        self._model = nn.Sequential(self.fe, 
                                    self.flatten_layer,
                                    self.classification_head)

        self._transform = self.fe.transform

def _contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    m = isinstance(module, nn.Linear)
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    return m and sub_m


def initialize_alexnet_classifier_from_config(configuration: Dict, seed: int) -> Tuple[AlexnetClassifierWrapper, Dict]:
    # add this command to make sure the same initialization is used for the same configuration in any run
    pu.seed_everything(seed=seed)

    keys = set(list(configuration.keys()))
    if not {"input_shape","num_classes","feature_extractor_blocks", "frozen_feature_extractor_blocks", "num_classification_layers"}.issubset(keys): 
        raise ValueError(f'one of the following keys is missing: {["input_shape","num_classes","feature_extractor_blocks", "frozen_feature_extractor_blocks", "num_classification_layers"]}')    

    temp_fe = afe.AlexNetFeatureExtractor(model_blocks=configuration['feature_extractor_blocks'], 
                                          frozen_model_blocks=configuration['frozen_feature_extractor_blocks'])

    if _contains_fc_layer(temp_fe):
        raise ValueError(f"Please make sure to initialize the feature extract such that there is not fully connected layers")

    # convert the input shape to tuple
    configuration['input_shape'] = tuple(configuration['input_shape'])

    if 'learning_rate' not in configuration:
        configuration['learning_rate'] = 10 ** -4 
    
    if 'num_vis_images' not in configuration: 
        configuration['num_vis_images'] = 3

    # make sure to pass the optimizer and scheduler through the 'str_to_cls_map' functionalities
    if 'optimizer_class' not in configuration:
        configuration['optimizer_class'] = torch.optim.Adam 
    else: 
        configuration['optimizer_class'] = scm.optimizer_str_to_cls(configuration['optimizer_class'])

    # the default is to pass no keywargs to the optimizer
    if 'optimizer_keyargs' not in configuration: 
        configuration['optimizer_keyargs'] = None
    else: 
        if not (isinstance(configuration['optimizer_keyargs'], Dict) or configuration['optimizer_keyargs'] is None):
            raise ValueError(f"The keyword arguments must of the 'Dict' shape. Found: {configuration['optimizer_keyargs']}")

    if 'scheduler_class' not in configuration:
        configuration['scheduler_class'] = torch.optim.lr_scheduler.ExponentialLR
    else: 
        sc = configuration['scheduler_class']
        if isinstance(sc, List):
            configuration['scheduler_class'] = [scm.scheduler_str_to_cls(c) for c in sc]
        else:
            configuration['scheduler_class'] = scm.scheduler_str_to_cls(configuration['scheduler_class'])

    if 'scheduler_keyargs' not in configuration: 
        configuration['scheduler_keyargs'] = {"gamma": 99}
    else: 
        # make sure that the field of 'scheduler_keyargs' is either a dictionary or a list of dictionaries
        skwargs = configuration['scheduler_keyargs']
        skwargs = skwargs if isinstance(sc, List) else [skwargs]        
        for c in skwargs:
            if not isinstance(c, Dict):
                raise ValueError(f"Keyword arguments must of the 'Dict' shape. Found: {configuration['scheduler_keyargs']}")


    arg_keys = [
                'num_epochs',
                "input_shape",
                "num_classes",
                "num_classification_layers", 
                "feature_extractor_blocks",
                "frozen_feature_extractor_blocks",
                'optimizer_class',
                'learning_rate',
                'scheduler_class',
                'num_vis_images', 
                'optimizer_keyargs',
                'scheduler_keyargs']
    
    if sorted(arg_keys) != sorted(list(configuration.keys())):
        raise ValueError(f"The configuration's keys are not the same as the expected ones. Missing: {set(arg_keys).difference(set(list(configuration.keys())))}"
                         f"\nExtra: {set(list(configuration.keys())).difference(set(arg_keys))}")     

    # # pass the non-keyword arguments independently
    args = {k: configuration[k] for k in arg_keys if k != 'num_epochs'}
    
    wrapper = AlexnetClassifierWrapper(**args)

    return wrapper, configuration
