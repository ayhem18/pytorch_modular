"""
This script contais the training code for the SimClrWrapper PL model
"""


import pytorch_lightning as L

from clearml import Task, Logger
from typing import Union, Optional, Tuple, List
from pathlib import Path
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import CombinedLoader

from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.shortcuts import P

from .simClrWrapper import ResnetSimClrWrapper
from .set_ds import _set_data

from .constants import (_TEMPERATURE, _OUTPUT_DIM, _OUTPUT_SHAPE, 
                        ARCHITECTURE_IMAGENETTE, ARCHITECTURE_FOOD_101, 
                        _NUM_FC_LAYERS, _DROPOUT,
                        TRACK_PROJECT_NAME) 


def _set_logging(use_logging: bool, 
                 run_name: str, 
                 return_task: bool, 
                 init_task:bool=True,
                 **kwargs) -> Union[Tuple[Task, Logger], Optional[Logger]]:

    if use_logging:
        # create a clearml task object
        if init_task:
            task = Task.init(project_name=TRACK_PROJECT_NAME,
                            task_name=run_name,
                            **kwargs
                            )
            logger = task.get_logger()
        else:
            task = Task.create(project_name=TRACK_PROJECT_NAME,
                            task_name=run_name,
                            # **kwargs
                            )
            logger = task.get_logger()
    else:
        logger = None
        task = None
    
    if return_task:
        return task, logger
    
    return logger


def _set_ckpnt_callback(val_dl: Optional[DataLoader], 
                        num_epochs:int, 
                        val_per_epoch:int,
                        log_dir: P) -> ModelCheckpoint:

    if val_dl is not None:
        # if the validation dataset is passed, make sure
        # the val_per_epoch argument is less than half the number of epochs
        # so that at least 2 checkpoints are considered     
        if num_epochs // 2 < val_per_epoch:
            raise ValueError("Passing a validation dataset while setting the 'val_per_epoch' less than the number of epochs will eventually raise an error !!!") 


    ckpnt_split = 'val' if val_dl is not None else 'train'
    # if the validation dataset is passed, then we need to have al least one validation epoch before checkpointing
    # otherwise update the checkpoint every epoch (we need the best training loss...)
    every_n_epochs = val_per_epoch if val_dl is not None else 1
    monitor = f"{ckpnt_split}_epoch_loss"

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=1, 
                                        monitor=monitor, # the checkpointing process might depend on either the train on the validation data
                                        mode='min', 
                                        every_n_epochs=every_n_epochs, 
                                        # save the checkpoint with the epoch and validation loss
                                        filename= ckpnt_split + '-{epoch:02d}-'+ '{' + monitor + ':06f}',
                                        # auto_insert_metric_name=True,
                                        save_on_train_epoch_end=(val_dl is None) # if there is a validation set then check after the val epoch
                                        )

    return checkpnt_callback


def train_simClr_wrapper(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        batch_size: int,

        log_dir: Union[str, Path],
        run_name: str,

        debug_augmentations: List,
        output_shape:Tuple[int, int]=None, 
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,

        use_logging: bool = True,

        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ) -> ResnetSimClrWrapper:    

    if output_shape is None:
        output_shape = _OUTPUT_SHAPE
    
    # process the loggin directory
    log_dir = dirf.process_path(log_dir, file_ok=False, dir_ok=True)

    # set the logger
    logger = _set_logging(use_logging=use_logging, 
                        run_name=run_name, 
                        return_task=False)

    simClr_train_dl, simClr_val_dl, debug_train_dl = _set_data(train_data_folder=train_data_folder,
                                                    val_data_folder=val_data_folder, 
                                                    dataset="imagenette",
                                                    output_shape=output_shape,
                                                    batch_size=batch_size, 
                                                    num_train_samples_per_cls=num_train_samples_per_cls,
                                                    num_val_samples_per_cls=num_val_samples_per_cls,
                                                    seed=seed)

    checkpnt_callback = _set_ckpnt_callback(val_dl=simClr_val_dl, 
                                            num_epochs=num_epochs, 
                                            val_per_epoch=val_per_epoch, 
                                            log_dir=log_dir)

    lr = 0.3 * (batch_size / 256) # according to the paper's formula

    # define the wrapper
    wrapper = ResnetSimClrWrapper(
                                  # model parameters
                                  input_shape=(3,) + output_shape, 
                                  output_dim=_OUTPUT_DIM,
                                  num_fc_layers=_NUM_FC_LAYERS,

                                  #logging parameters
                                  logger=logger,
                                  log_per_batch=3,
                                  val_per_epoch=val_per_epoch,
                                  # loss parameters 
                                  temperature=_TEMPERATURE,
                                  debug_loss=True,

                                  # learning 
                                  lrs=lr,
                                  num_epochs=num_epochs,
                                  num_warmup_epochs=num_warmup_epochs,

                                  debug_embds_vs_trans=True, 
                                  debug_transformations=debug_augmentations,

                                  # more model parameters
                                  dropout=_DROPOUT,
                                  architecture=ARCHITECTURE_IMAGENETTE,
                                  freeze=False
                                  )

    # get the default device
    device = pu.get_default_device()

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu',     ## I don't have acess to TPUs at the time of writing this code)))
                        devices=1, # not sure of the aftermath of setting this parameter to 'auto' (1 is the safest choice for now)
                        logger=False, # since I am not using any of the supported logging options, logger=False + using the self.log function would do the job...
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=1 if len(simClr_train_dl) < 10 else 5,
                        callbacks=[checkpnt_callback])


    trainer.fit(model=wrapper,
                
                train_dataloaders=simClr_val_dl,
                val_dataloaders=CombinedLoader([simClr_val_dl, debug_train_dl], 
                                                    mode='sequential'), 
                )

    return wrapper


