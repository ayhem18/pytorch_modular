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


# THOSE VALUES WERE SET AFTER TUNING THE MODEL
_NUM_FC_LAYERS = 3
_DROPOUT = 0.2

# those are chosen manually
_OUTPUT_DIM = 128
_TEMPERATURE = 0.5

# 101 IS DEFINITELY AN OVERKILL
ARCHITECTURE = 50

def train_simClr_wrapper(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        output_shape:Tuple[int, int], 

        num_epochs: int, 
        batch_size: int,

        log_dir: Union[str, Path],
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,

        use_logging: bool = True,

        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ):    
    # process the loggin directory
    log_dir = dirf.process_path(log_dir)

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

    lr = 0.3 * (batch_size / 256) # according to the paper's formula

    # define the wrapper
    wrapper = ResnetSimClrWrapper(input_shape=output_shape, 
                                  output_dim=_OUTPUT_DIM,
                                  num_fc_layers=_NUM_FC_LAYERS,

                                  log_per_batch=10,
                                  temperature=_TEMPERATURE,
                                  debug_loss=True,
                                  lrs=lr,
                                  num_epochs=num_epochs,
                                  num_warmup_epochs=num_warmup_epochs,
                                  dropout=_DROPOUT,

                                  architecture=ARCHITECTURE,
                                  freeze=False
                                  )


    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=1, 
                                        monitor="val_loss", # the value of the monitor variable is tied to the logging process...
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='val-{epoch:02d}-{val_loss:06f}')


    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu', ## I don't have acess to TPUs at the time of writing this code)))
                        devices=1,
                        logger=False, # disable logging until figuring out how to combine ClearML with PytorchLightning 
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=val_per_epoch,
                        log_every_n_steps=1 if len(train_dl) < 10 else 10,
                        callbacks=[checkpnt_callback])


    trainer.fit(model=wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )
