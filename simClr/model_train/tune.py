"""
This script contains the implementation of the tuning process
"""
import wandb

from pathlib import Path
from typing import Union, Dict, Optional, Tuple

from mypt.code_utilities import directories_and_files as dirf

from .train import run_pipeline
from .models.resnet.model import ResnetSimClr

WANDB_PROJECT_NAME="SimClr"

def _sweep_function( 
          train_data_folder:Union[str, Path],
          val_data_folder:Optional[Union[str, Path]],

          num_epochs: int, 
          batch_size: int,

          temperature: float,
          ckpnt_dir: Union[str, Path],
          val_per_epoch: int = 3,
          seed:int = 69,        
         ):
    
    # extract the number of fully connected layers form the sweep configuration
    num_fc_layers = wandb.config.num_fc_layers
    lr = wandb.config.lr    

    model = ResnetSimClr(input_shape=(3, 224, 224), output_dim= 256, num_fc_layers=3, dropout=0.3)

    # train the model
    pass

def tune(train_data_folder: Union[str, Path],
         val_data_folder: Optional[Union[str, Path]],
         lr_options: Dict,
         num_layers_options: Dict,
         objective: str,
         tune_method:str,
        ):

    if objective not in ['train_loss', 'val_loss']:
        raise NotImplementedError(f"objective not currently supported. Found: {objective}, supporting only: {['train_loss', 'val_loss']}")
    
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=WANDB_PROJECT_NAME)

    # prepare the wandb configuration
    sweep_config = {
        "name": "cbm_sweep", 
        "metric": {"name": "val_loss", "goal": "minimize"}, 
        "method": tune_method, 
        "parameters": {"num_fc_layers": num_layers_options, "lr": lr_options},
    }

    log_dir = dirf.process_path(os.path.join(log_dir, f'sweep_{os.path.splitext(os.path.basename(cbm_configuration_path))[0]}'))
    
    sweep_id = wandb.sweep(sweep=sweep_config, 
                    project=WANDB_PROJECT_NAME)


    # let's define the object that the sweep agent will run
    sweep_function_object = partial(_sweep_function, 
                                    log_dir=log_dir, 
                                    sweep_config=sweep_config,
                                    seed=seed,
                                    dir_train=train_dir, 
                                    dir_val=val_dir, 
                                    concepts=concepts, 
                                    epoch_per_sweep=epoch_per_sweep,
                                    )

    # run the sweep
    wandb.agent(sweep_id, 
            function=sweep_function_object,
            count=sweep_count)
    
