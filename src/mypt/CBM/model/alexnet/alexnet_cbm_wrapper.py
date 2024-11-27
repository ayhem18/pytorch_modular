"""
This script contains the implementation of the wrapper of CBM with Resnet backbone.
"""
import os, torch
# from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, List, Any


WANDB_PROJECT_NAME = 'CBM-UDA'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from . import alexnet_cbm as acbm
from ..backbone_cbm_wrapper import BackboneCbmWrapper
from ..loss import CBMLoss, BinaryCBMLoss
from ....backbones import alexnetFeatureExtractor as afe
from ....code_utilities import str_class_map as scm
from ....code_utilities import pytorch_utilities as pu


class AlexnetCbmWrapper(BackboneCbmWrapper):
    def __init__(self, 
                input_shape: Tuple,
                num_concepts: int, 
                num_classes: int,   

                num_concept_layers: int,
                num_classification_layers: int,

                optimizer_class: callable,
                loss: callable,

                feature_extractor_blocks: Union[str, List[str], int, List[int]] = 'conv_block_adapool',
                frozen_feature_extractor_blocks: Union[str, List[str], int, List[int]] = 'conv_block_adapool',                
                concept_projection_dropout: float = None,
                
                learning_rate: Optional[float] = 10 ** -4,
                loss_coefficient: float = 1,                 
                num_vis_images: int = 3,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,
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
                        scheduler_keyargs=scheduler_keyargs                
                        )
        # if either self.scheduler_args or self.scheduler_class are lists: basically different rl schedulers for the backbone and
        # the classification head, then we will have 2 optimizers for each component and we need to activate self.automatic_optimizer        
        if isinstance(self.scheduler_args, List) or isinstance(self.scheduler_class, List):
            self.automatic_optimization = False

        self._fe = afe.AlexNetFeatureExtractor(model_blocks=feature_extractor_blocks, 
                                               frozen_model_blocks=frozen_feature_extractor_blocks)

        # create the inner model
        self._model = acbm.AlexnetCBM(input_shape=input_shape, 
                                num_concepts=num_concepts, 
                                num_classes=num_classes,
                                feature_extractor=self._fe,
                                num_classification_layers=num_classification_layers,
                                num_concept_layers=num_concept_layers, 
                                concept_projection_dropout=concept_projection_dropout
                                )
        
        # save the transform used with the model
        self._transform = self._fe.transform 



def initialize_alexnet_cbmw_from_config(config: Dict, seed: int) -> Tuple[AlexnetCbmWrapper, Dict]:

    pu.seed_everything(seed=seed)
    # copy the passed to avoid modifying it
    configuration = config.copy()

    keys = set(list(configuration.keys()))
    
    necessary_fields = {"input_shape",
                        "num_concepts",
                        "num_classes", # these 3 arguments are needed for any CBM model

                        "num_concept_layers", # needed for the concept bottleneck
                        "num_classification_layers", # needed for the classification head

                        "feature_extractor_blocks",                
                        "frozen_feature_extractor_blocks", # these 2 fields are needed for AlexNet specifically               

                        "similarity", # needed to define the loss 
                        "representation"}
    
    if not necessary_fields.issubset(keys):
        missing_fields = necessary_fields.difference(keys)
        raise ValueError(f'some of the necessary keys are missing, exaclty: {missing_fields}')    

    # convert the input shape to tuple
    configuration['input_shape'] = tuple(configuration['input_shape'])

    if 'concept_projection_dropout' not in configuration:
        configuration['concept_projection_dropout'] = None

    if 'learning_rate' not in configuration:
        configuration['learning_rate'] = 10 ** -4 

    if 'loss_coefficient' not in configuration:
        configuration['loss_coefficient'] = 1

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

    arg_keys = [
                'num_epochs',
                "input_shape",
                "num_concepts",
                "num_classes",
                "num_concept_layers",
                "num_classification_layers", 
                
                "feature_extractor_blocks", 
                "frozen_feature_extractor_blocks",
                
                'optimizer_class',
                'learning_rate',
                'scheduler_class',
                'num_vis_images', 
                'scheduler_keyargs',
                'optimizer_keyargs', 
                'concept_projection_dropout', 
                'loss_coefficient', 
                "similarity", 
                "representation", 
                'loss']
    
    if sorted(arg_keys) != sorted(list(configuration.keys())):
        raise ValueError(f"The configuration has extra keys: {set(list(configuration.keys())).difference(set(arg_keys))}"
                         f"\nMissing keys: {set(arg_keys).difference(set(list(configuration.keys())))}")

    # # pass the non-keyword arguments independently
    # args = {k: configuration[k] for k in arg_keys if k != 'scheduler_keyargs'}
    
    # remove 'num_epochs' from the configuration
    args = {k: v for k, v in configuration.items() if k not in {'num_epochs', 'representation', 'similarity'}}

    wrapper = AlexnetCbmWrapper(**args)

    return wrapper, configuration
