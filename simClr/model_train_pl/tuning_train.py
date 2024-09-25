"""
This script contains the code for tuning the Pytorch Lightning module
"""
import os, pytorch_lightning as L


from typing import Union, Optional, Tuple, List
from pathlib import Path

from clearml import Task, Logger
from clearml.automation.optimization import HyperParameterOptimizer

from clearml.automation import DiscreteParameterRange, LogUniformParameterRange
from clearml.automation.optuna import OptimizerOptuna

from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf

from .set_ds import _set_data_tune
from .train import  _set_logging, _set_ckpnt_callback
from .simClrWrapper import ResnetSimClrWrapper
from .constants import (_TEMPERATURE, _OUTPUT_DIM, _OUTPUT_SHAPE, 
                        ARCHITECTURE_IMAGENETTE, ARCHITECTURE_FOOD_101, 
                        TRACK_PROJECT_NAME, TUNING_TASK_RUN_NAME) 

def tune_template_task_function(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        batch_size: int,

        parent_log_dir: Union[str, Path], # parent_log_dir where all the logs from the different sub-folders will be saved
        tune_exp_number: int,
        output_shape:Tuple[int, int]=None, 
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,
        set_up: bool = False,
    ):

    if output_shape is None:
        output_shape = _OUTPUT_SHAPE

    # and since the 'sub_exp_logger' is actually connected to the 'task' object, ClearML should be able to capture the desired metrics
    task, sub_exp_logger = _set_logging(use_logging=True, 
                                        run_name=f'{TUNING_TASK_RUN_NAME}_{tune_exp_number}', 
                                        return_task=True,
                                        init_task=False, 
                                        auto_connect_arg_parser=False, 
                                        auto_connect_frameworks=False, 
                                        auto_connect_streams=False
                                        )
    
    
    # connect the task to the 'num_fc_layers' and 'dropout' parameters
    args = {'num_fc_layers': None, 'dropout': None}
    
    if set_up:
        args.update({
                "train_data_folder":train_data_folder,                 
                "val_data_folder":val_data_folder,
                "dataset": dataset,

                "num_epochs": num_epochs, 
                "batch_size": batch_size,

                "parent_log_dir": parent_log_dir,
                "tune_exp_number": tune_exp_number,
                "output_shape":output_shape, 
                "num_warmup_epochs":num_warmup_epochs,
                "val_per_epoch": val_per_epoch,
                "seed":seed,
                'set_up': True
                })

    task.connect(args)
    
    if args['set_up']:
        return 


    # set the data
    train_dl, val_dl = _set_data_tune(train_data_folder=args['train_data_folder'],
                                val_data_folder=args['val_data_folder'], 
                                dataset=args['dataset'],
                                output_shape=args['output_shape'],
                                batch_size=args['batch_size'], 
                                seed=args['seed'])
    
    initial_lr = 0.3 * (args['batch_size'] / 256) # according to the paper's formula
    
    
    # the exact loging directory depends on the number of experiments conducted so far
    sub_exp_log_dir = dirf.process_path(os.path.join(args['parent_log_dir'], f'sub_exp_{len(os.listdir(args["parent_log_dir"])) + 1}'))


    checkpnt_callback = _set_ckpnt_callback(val_dl=args['val_dl'], 
                                            num_epochs=args['num_epochs'], 
                                            val_per_epoch=args['val_per_epoch'], 
                                            log_dir=args['sub_exp_log_dir'])

    wrapper = ResnetSimClrWrapper(
                                # model parameters
                                input_shape=(3,) + args['output_shape'], 
                                output_dim=_OUTPUT_DIM,
                                num_fc_layers=args['num_fc_layers'],

                                #logging parameters
                                logger=sub_exp_logger,
                                log_per_batch=3,
                                
                                # loss parameters 
                                temperature=_TEMPERATURE,
                                debug_loss=True,

                                # learning 
                                lrs=initial_lr,
                                num_epochs=args['num_epochs'],
                                num_warmup_epochs=args['num_warmup_epochs'],

                                # more model parameters
                                dropout=args['dropout'],
                                architecture=ARCHITECTURE_FOOD_101 if args['dataset']=='food101' else ARCHITECTURE_IMAGENETTE,
                                freeze=False
                                )

    # get the default device
    device = pu.get_default_device()

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu',
                        devices=1, 
                        logger=False, 
                        default_root_dir=sub_exp_log_dir,
                        max_epochs=args['num_epochs'],
                        check_val_every_n_epoch=args['val_per_epoch'],
                        log_every_n_steps=1 if len(train_dl) < 10 else 10,
                        callbacks=[checkpnt_callback])


    trainer.fit(model=wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                )

    
