"""
This script contains the implementation of Pytorch Lightning wrapper around a standard classifier with a Resnet backbone.
"""

import torch
from torch import nn
from typing import Union, Tuple, Optional, List, Dict
from copy import deepcopy

from ..backbone_classifier_wrapper import BackboneClassifierWrapper
from ...backbones import resnetFeatureExtractor as rfe
from ...linearBlocks import classification_head as ch
from ...dimensions_analysis import dimension_analyser as da
from ...code_utilities import str_class_map as scm 
from ...code_utilities import pytorch_utilities as pu 
from ...swag.posteriors.swag import SWAG

class ResnetClassifierWrapper(BackboneClassifierWrapper):
    def __init__(self, 
                input_shape: Tuple,
                num_classes: int,   
                num_classification_layers: int, 

                optimizer_class: callable,

                fe_num_blocks: int,  # the number of blocks to keep in the feature extractor
                freeze_fe: Union[bool, int] = True, # the number of layers to freeze
                add_global_average: bool = True,
                architecture: int = 50,
                freeze_layers: bool = True,
                add_fc: bool = False,

                learning_rate: Optional[float] = 10 ** -4,
                num_vis_images: int = 3,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_class: Optional[callable] = None,
                scheduler_keyargs: Optional[Dict] = None,   
                dropout: Optional[float] = None,
                swag: bool = False
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
        
        # initialize the feature extractor from the passed arguments
        self.fe = rfe.ResNetFeatureExtractor(num_layers=fe_num_blocks, 
                                        add_global_average=add_global_average, 
                                        architecture=architecture,
                                        freeze=freeze_fe, 
                                        freeze_layers=freeze_layers, 
                                        # make sure to add the last layer only if the number of classes is the same
                                        # as the one Resnet was initial trained on.
                                        add_fc=(add_fc and num_classes == rfe.DEFAUL_NUM_CLASSES))

        # if the 'add_fc' argument is set to True and the number of classes is the default one, then use resnetFeature extractor as it is
        if add_fc and num_classes == rfe.DEFAUL_NUM_CLASSES: 
            self._model = self.fe
        else:
            # define a dimension analyser: a class that calculates the shape of the output 
            # of a nn.Module given an input shape (without running the model)
            dim_analyser = da.DimensionsAnalyser(method='static')

            # calculate the number of features passed to the classification head.
            _, in_features = dim_analyser.analyse_dimensions(input_shape=(10,) + input_shape, net=nn.Sequential(self.fe, nn.Flatten()))

            # initialize the classification head
            self.classification_head = ch.ExponentialClassifier(num_classes=num_classes, 
                                                                in_features=in_features, 
                                                                num_layers=num_classification_layers,
                                                                dropout=dropout)
            
            self.flatten_layer = nn.Flatten()

            # set the model regardless of the swag parameter
            self._model = nn.Sequential(self.fe, 
                                        self.flatten_layer,
                                        self.classification_head)

            # depending on the swag parameter
            if swag: 
                self._swag_model = SWAG(base_model=self._model, 
                                    no_cov_mat=False # setting no_cov_mat to True leads to errors during sampling
                                    )
            else:
                self._swag_model = None

        # save the transform used with the model
        self._transform = self.fe.transform 


def initialize_resnet_classifier_from_config(config: Dict, 
                                             seed: int) -> Tuple[ResnetClassifierWrapper, Dict]:
    configuration = config.copy()
    # add this command to make sure the same initialization is used for the same configuration in any run
    pu.seed_everything(seed=seed)

    keys = set(list(configuration.keys()))
    if not {"input_shape","num_classes","fe_num_blocks", "num_classification_layers"}.issubset(keys): 
        raise ValueError(f'one of the following keys is missing: {["input_shape","num_classes","fe_num_blocks", "num_classification_layers"]}')    

    # convert the input shape to tuple
    configuration['input_shape'] = tuple(configuration['input_shape'])

    if 'learning_rate' not in configuration:
        configuration['learning_rate'] = 10 ** -4 

    if 'freeze_fe' not in configuration:
        configuration['freeze_fe'] = True 

    if 'add_global_average' not in configuration:
        configuration['add_global_average'] = True 

    if 'architecture' not in configuration:
        configuration['architecture'] = 50  

    if 'add_fc' not in configuration:
        configuration['add_fc'] = False # the default is not to use the last linear layer

    if 'freeze_layers' not in configuration:
        configuration['freeze_layers'] = True # the default is to freeze layers and not residual blocks
    
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


    # make sure if 'add_fc' is set to True then 'num_classification_layers' is set to '1'
    if configuration['add_fc'] and configuration['num_classification_layers'] != 1:
        raise ValueError((f"if the 'add_fc' is set to True, then 'num_classification_layers' must be set to '1'.\n" 
                         f"Found: {configuration['num_classification_layers']}"))

    if 'swag' not in configuration:
        configuration['swag'] = False

    # the default is to use no dropout
    if 'dropout' not in configuration:
        configuration['dropout'] = None

    arg_keys = [
                'num_epochs',
                "input_shape",
                "num_classes",
                "fe_num_blocks", 
                "num_classification_layers", 
                'optimizer_class',
                'learning_rate',
                'freeze_fe',
                'add_global_average',
                'add_fc', 
                'freeze_layers',
                'scheduler_class',
                'num_vis_images', 
                'architecture', 
                'optimizer_keyargs',
                'scheduler_keyargs', 
                'swag',
                'dropout'
                ]
    
    if sorted(arg_keys) != sorted(list(configuration.keys())):
        raise ValueError(f"The configuration's keys are not the same as the expected ones. Missing: {set(arg_keys).difference(set(list(configuration.keys())))}"
                         f"\nExtra: {set(list(configuration.keys())).difference(set(arg_keys))}")     

    # # pass the non-keyword arguments independently
    args = {k: configuration[k] for k in arg_keys if k != 'num_epochs'}
    
    wrapper = ResnetClassifierWrapper(**args)

    return wrapper, configuration
