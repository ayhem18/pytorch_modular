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

def tune_main_function(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_jobs: int,
        num_epochs_per_job: int, 
        batch_size: int,

        parent_log_dir: Union[str, Path], # parent_log_dir where all the logs from the different sub-folders will be saved

        output_shape:Tuple[int, int]=None, 
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,
    ):

    if output_shape is None:
        output_shape = _OUTPUT_SHAPE

    # set the data
    train_dl, val_dl = _set_data_tune(train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder, 
                                dataset=dataset,
                                output_shape=output_shape,
                                batch_size=batch_size, 
                                seed=seed)
    
    initial_lr = 0.3 * (batch_size / 256) # according to the paper's formula


    # define a function as a the tuning task
    def tune_task_function(
        num_fc_layers: int, 
        dropout: float,
        set_up: bool=False
    ):
        if set_up:
            # create the the task with the given run_name
            task = Task.init(project_name=TRACK_PROJECT_NAME, task_name=TUNING_TASK_RUN_NAME)
            Task
            task.connect('num_fc_layers')
            return
        
        # the exact loging directory depends on the number of experiments conducted so far
        sub_exp_log_dir = dirf.process_path(os.path.join(parent_log_dir, f'sub_exp_{len(os.listdir(parent_log_dir)) + 1}'))

        # set the logging
        sub_exp_logger = _set_logging(
                                    use_logging=True, 
                                    task_name=TUNING_TASK_RUN_NAME, # use the same run_name as in setup mode
                                    # pass this argument as per the instructions of the ClearMl tutorial: 
                                    # https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt/
                                    reuse_last_task_id=False  
                                    )
        
        # set the checkpoint 

        checkpnt_callback = _set_ckpnt_callback(val_dl=val_dl, 
                                                num_epochs=num_epochs_per_job, 
                                                val_per_epoch=val_per_epoch, 
                                                log_dir=sub_exp_log_dir)


        wrapper = ResnetSimClrWrapper(
                                    # model parameters
                                    input_shape=(3,) + output_shape, 
                                    output_dim=_OUTPUT_DIM,
                                    num_fc_layers=num_fc_layers,

                                    #logging parameters
                                    logger=sub_exp_logger,
                                    log_per_batch=3,
                                    
                                    # loss parameters 
                                    temperature=_TEMPERATURE,
                                    debug_loss=True,

                                    # learning 
                                    lrs=initial_lr,
                                    num_epochs=num_epochs_per_job,
                                    num_warmup_epochs=num_warmup_epochs,

                                    # more model parameters
                                    dropout=dropout,
                                    architecture=ARCHITECTURE_FOOD_101 if dataset=='food101' else ARCHITECTURE_IMAGENETTE,
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
                            max_epochs=num_epochs_per_job,
                            check_val_every_n_epoch=val_per_epoch,
                            log_every_n_steps=1 if len(train_dl) < 10 else 10,
                            callbacks=[checkpnt_callback])


        trainer.fit(model=wrapper,
                    train_dataloaders=train_dl,
                    val_dataloaders=val_dl,
                    )

    # call the tuning function once to initialize the tuning task
    tune_task_function(None, None, set_up=True)

    # define an optimizer
    tune_optimizer = HyperParameterOptimizer(
        base_task_id=Task.get_task(project_name=TRACK_PROJECT_NAME, task_name=TUNING_TASK_RUN_NAME).id, 

        hyper_parameters=[
            LogUniformParameterRange(name='dropout', min_value='0.1', max_value='1'),
            DiscreteParameterRange(name='num_fc_layers', values=list(range(2, 6))), # from 2 to 5 fully connected layers 
            DiscreteParameterRange(name='set_up', values=[False]), # from 2 to 5 fully connected layers 
        ],
        # this is the objective metric we want to maximize/minimize
        objective_metric_title='val_epoch_loss',
        objective_metric_series='val_epoch_loss',
        # now we decide if we want to maximize it or minimize it (accuracy we maximize)
        objective_metric_sign='min',
        max_number_of_concurrent_tasks=1,

        optimizer_class=OptimizerOptuna,
        # If specified all Tasks created by the HPO process will be created under the `spawned_project` project
        spawn_project=None,  
        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=3,  # 5,

        max_iteration_per_job=1, # the exact value of this parameter is not clear yet...
        total_max_jobs=num_jobs
    )

    # run the optimizer
    tune_optimizer.start_locally()

    # make sure to call the 'wait' function as the experiments are running on the background 
    # and not the main process...
    tune_optimizer.wait()

    # not setting the time limit for now
    tune_optimizer.stop()
    
