"""
This script contains the implementation of the wrapper of CBM with Resnet backbone.
"""
import os, torch
# from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, List
from copy import deepcopy

WANDB_PROJECT_NAME = 'CBM-UDA'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from . import resnet_cbm as rcbm
from ..backbone_cbm_wrapper import BackboneCbmWrapper
from ..loss import CBMLoss, BinaryCBMLoss
from ....backbones import resnetFeatureExtractor as rfe
from ....code_utilities import str_class_map as scm
from ....code_utilities import pytorch_utilities as pu
from ....swag.posteriors.swag import SWAG

class ResnetCbmWrapper(BackboneCbmWrapper):
    def __init__(self, 
                input_shape: Tuple,
                num_concepts: int, 
                num_classes: int,   
                
                fe_num_blocks: int,  # the number of blocks to keep in the feature extractor
                num_concept_layers: int,
                num_classification_layers: int, 

                optimizer_class: callable,
                loss: callable,

                freeze_fe: Union[bool, int] = True, # the number of layers to freeze
                add_global_average: bool = True,
                architecture: int = 50,
                freeze_layers: bool = True, 
                
                dropout: float = None,
                learning_rate: Optional[float] = 10 ** -4,
                loss_coefficient: float = 0.5,                 
                num_vis_images: int = 3,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,
                swag: bool = False,
                ):
        # parent class constructor
        super().__init__(
                        input_shape=input_shape, 
                        num_classes=num_classes, 
                        num_concepts=num_concepts,

                        optimizer_class=optimizer_class,
                        learning_rate=learning_rate,
                        scheduler_class=scheduler_class,
                        
                        loss = loss,
                        loss_coefficient=loss_coefficient,
                        num_vis_images=num_vis_images,

                        optimizer_keyargs=optimizer_keyargs,
                        scheduler_keyargs=scheduler_keyargs,          
                        )
        # if either self.scheduler_args or self.scheduler_class are lists: basically different rl schedulers for the backbone and
        # the classification head, then we will have 2 optimizers for each component and we need to activate self.automatica_optimizer        
        if isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List):
            self.automatic_optimization = False

        self._fe = rfe.ResNetFeatureExtractor(num_layers=fe_num_blocks, 
                                          add_global_average=add_global_average, 
                                          architecture=architecture, 
                                          freeze=freeze_fe, 
                                          freeze_layers=freeze_layers, 
                                          add_fc=False)

        # set the self._model attribute regardless
        self._model = rcbm.ResnetCBM(input_shape=input_shape, 
                                num_concepts=num_concepts, 
                                num_classes=num_classes,
                                feature_extractor=self._fe,
                                num_classification_layers=num_classification_layers,
                                num_concept_layers=num_concept_layers, 
                                dropout=dropout
                                )

        if swag:
            self._swag_model = SWAG(self._model, 
                                    no_cov_mat=False # setting no_cov_mat to True leads to errors during sampling
                                    )
        else:
            self._swag_model = None
        
        # save the transform used with the model
        self._transform = self._fe.transform 

def initialize_resnet_cbmw_from_config(config: Dict, seed: int) -> Tuple[ResnetCbmWrapper, Dict]:

    pu.seed_everything(seed=seed)
    # copy the passed to avoid modifying it
    configuration = config.copy()

    keys = set(list(configuration.keys()))
    
    necessary_fields = {"input_shape",
            "num_concepts",
            "num_classes",
            "fe_num_blocks", 
            "num_concept_layers",
            "num_classification_layers", 
            "similarity", 
            "representation"}
    
    if not necessary_fields.issubset(keys):
        missing_fields = necessary_fields.difference(keys)
        raise ValueError(f'some of the necessary keys are missing, exaclty: {missing_fields}')    

    # convert the input shape to tuple
    configuration['input_shape'] = tuple(configuration['input_shape'])

    if 'dropout' not in configuration:
        configuration['dropout'] = None

    if 'learning_rate' not in configuration:
        configuration['learning_rate'] = 10 ** -4 

    if 'freeze_fe' not in configuration:
        configuration['freeze_fe'] = True 

    if 'add_global_average' not in configuration:
        configuration['add_global_average'] = True 

    if 'architecture' not in configuration:
        configuration['architecture'] = 50  

    if 'freeze_layers' not in configuration:
        configuration['freeze_layers'] = True # the default is to freeze layers and not residual blocks

    if 'loss_coefficient' not in configuration:
        configuration['loss_coefficient'] = 0.5

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
        skwargs = skwargs if (isinstance(skwargs, List)) else [skwargs]        
        for c in skwargs:
            if not isinstance(c, Dict):
                raise ValueError(f"Keyword arguments must of the 'Dict' shape. Found: {configuration['scheduler_keyargs']}")


    # make sure to specify the loss
    if 'loss' not in configuration:
        # then we use the default loss
        configuration['loss'] =  CBMLoss
    else:
        configuration['loss'] = scm.loss_str_to_cls(configuration['loss'])

    if configuration['representation'] == 3 and configuration['loss'] != BinaryCBMLoss:
        raise ValueError(f"If the model is to be trained with the 3rd representation, then the loss must be binary !!!")

    # add the 'swag' parameter
    if 'swag' not in configuration: 
        configuration['swag'] = False

    arg_keys = [
                'num_epochs',
                "input_shape",
                "num_concepts",
                "num_classes",
                "fe_num_blocks", 
                "num_concept_layers",
                "num_classification_layers", 
                'optimizer_class',
                'learning_rate',
                'freeze_fe',
                'add_global_average',
                'freeze_layers', 
                'scheduler_class',
                'num_vis_images', 
                'architecture', 
                'scheduler_keyargs',
                'optimizer_keyargs', 
                'dropout', 
                'loss_coefficient', 
                "similarity", 
                "representation", 
                'loss', 
                'swag']
    
    if sorted(arg_keys) != sorted(list(configuration.keys())):
        raise ValueError(f"The configuration has extra keys: {set(list(configuration.keys())).difference(set(arg_keys))}"
                         f"\nMissing keys: {set(arg_keys).difference(set(list(configuration.keys())))}")
    
    # remove 'num_epochs' from the configuration
    args = {k: v for k, v in configuration.items() if k not in {'num_epochs', 'representation', 'similarity'}}

    wrapper = ResnetCbmWrapper(**args)

    return wrapper, configuration


