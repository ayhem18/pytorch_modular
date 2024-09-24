"""
This script contais the training code for the SimClrWrapper PL model
"""


import pytorch_lightning as L

from clearml import Task, Logger
from typing import Union, Optional, Tuple, List
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint

from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf

from .simClrWrapper import ResnetSimClrWrapper
from .set_ds import _set_data


# project name on clearml
TRACK_PROJECT_NAME = 'simClr'

# These VALUES WERE SET AFTER TUNING THE MODEL
_NUM_FC_LAYERS = 3
_DROPOUT = 0.2

# these are chosen manually
_OUTPUT_DIM = 128
_TEMPERATURE = 0.5
_OUTPUT_SHAPE = (200, 200)

# 101 IS DEFINITELY AN OVERKILL, so let's see how it goes...
ARCHITECTURE = 50

def train_simClr_wrapper(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        batch_size: int,

        log_dir: Union[str, Path],
        run_name: str,

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

    # get the default device
    device = pu.get_default_device()

    # DATA: 
    train_dl, val_dl = _set_data(train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder, 
                                dataset=dataset,
                                output_shape=output_shape,
                                batch_size=batch_size, 
                                num_train_samples_per_cls=num_train_samples_per_cls,
                                num_val_samples_per_cls=num_val_samples_per_cls,
                                seed=seed)

    if val_dl is not None:
        # if the validation dataset is passed, make sure
        # the val_per_epoch argument is less than half the number of epochs
        # for Pytorch lightning to consider at least 2 validation epochs        
        if num_epochs // 2 < val_per_epoch:
            raise ValueError("Passing a validation dataset while setting the 'val_per_epoch' less than the number of epochs will eventually raise an error !!!") 

    lr = 0.3 * (batch_size / 256) # according to the paper's formula

    if use_logging:
        # create a clearml task object
        task = Task.init(project_name=TRACK_PROJECT_NAME,
                         task_name=run_name,
                         )
        logger = task.get_logger()
    else:
        logger = None
        
    # define the wrapper
    wrapper = ResnetSimClrWrapper(
                                  # model parameters
                                  input_shape=(3,) + output_shape, 
                                  output_dim=_OUTPUT_DIM,
                                  num_fc_layers=_NUM_FC_LAYERS,

                                  #logging parameters
                                  logger=logger,
                                  log_per_batch=3,
                                  
                                  # loss parameters 
                                  temperature=_TEMPERATURE,
                                  debug_loss=True,

                                  # learning 
                                  lrs=lr,
                                  num_epochs=num_epochs,
                                  num_warmup_epochs=num_warmup_epochs,

                                  # more model parameters
                                  dropout=_DROPOUT,
                                  architecture=ARCHITECTURE,
                                  freeze=False
                                  )

    ckpnt_split = 'val' if val_dl is not None else 'train'

    # if the validation dataset is passed, then we need to have al least on validation epoch before checkpointing
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


    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu',     ## I don't have acess to TPUs at the time of writing this code)))
                        devices=1, # not sure of the aftermath of setting this parameter to 'auto' (1 is the safest choiceTh)
                        logger=False, # disable logging until figuring out how to combine ClearML with PytorchLightning 
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=val_per_epoch,
                        log_every_n_steps=1 if len(train_dl) < 10 else 10,
                        callbacks=[checkpnt_callback])


    trainer.fit(model=wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                )

    return wrapper


