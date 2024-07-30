import wandb, torch, os

import torchvision.transforms as tr

from time import sleep
from typing import Union, Optional, Tuple, List
from pathlib import Path

from tqdm import tqdm
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from mypt.losses.simClrLoss import SimClrLoss
from mypt.data.datasets.parallel_aug_ds import ParallelAugDs
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.schedulers.annealing_lr import AnnealingLR

from .train_per_epoch import train_per_epoch, validation_per_epoch
from .models.resnet.model import ResnetSimClr

_DEFAULT_DATA_AUGS = [tr.RandomVerticalFlip(p=1), 
                      tr.RandomHorizontalFlip(p=1), 
                      tr.RandomRotation(degrees=15),
					  tr.RandomErasing(p=1, scale=(0.05, 0.15)),
					  tr.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),
                      tr.GaussianBlur(kernel_size=(5, 5)),
                      ]

# although the normalization was part of the original data augmentations, the normalized image loses a lot of its semantic meaning after normalization.
_UNIFORM_DATA_AUGS = []#tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    

_WANDB_PROJECT_NAME = "SimClr"

# let's split the training pipeline into several functions
def _set_data(train_data_folder: Union[str, Path],
             val_data_folder: Optional[Union[str, Path]],
             batch_size:int,
             output_shape: Tuple[int, int] = None,
             seed:int=69) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    if output_shape is None:
        output_shape = (224, 224)

    train_data_folder = dirf.process_path(train_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          condition=dirf.image_directory,
                                          error_message="The data diretory is expected to contain only image data"
                                          )

    train_ds = ParallelAugDs(root=train_data_folder, 
                       output_shape=output_shape, 
                       augs_per_sample=2, 
                       sampled_data_augs=_DEFAULT_DATA_AUGS,
                       uniform_data_augs=_UNIFORM_DATA_AUGS)

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          condition=dirf.image_directory,
                                          error_message="The data diretory is expected to contain only image data"
                                          )


        val_ds = ParallelAugDs(root=val_data_folder, 
                        output_shape=output_shape, 
                        augs_per_sample=2, 
                        data_augs=_DEFAULT_DATA_AUGS,
                        uniform_data_augs=_UNIFORM_DATA_AUGS)
    else:
        val_ds = None

    ## data loaders
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         warning=False # the num_workers=0 is deliberately set to 0  
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=train_ds, 
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            )
    else:
        val_dl = None

    return train_dl, val_dl


def _set_optimizer(model: ResnetSimClr, 
                  lrs: Union[Tuple[float], float], 
                  num_epochs: int) -> Tuple[SGD, AnnealingLR]:

    if isinstance(lrs, float):
        lr1, lr2 = lrs, 10 * lrs
    elif isinstance(lrs, List) and len(lrs) == 2:
        lr1, lr2 = lrs

    elif isinstance(lrs, List) and len(lrs) == 1:
        return _set_optimizer(model, lrs[0])
    else:
        raise ValueError(f"The current implementation supports at most 2 learning rates. Found: {len(lrs)} learning rates")


    # set the optimizer
    optimizer = SGD(params=[{"params": model.fe.parameters(), "lr": lr1}, # using different learning rates with different components of the model 
                            {"params": model.flatten_layer.parameters(), "lr": lr1},
                            {"params": model.ph.parameters(), "lr": lr2}
                            ])

    lr_scheduler = AnnealingLR(optimizer=optimizer, 
                               num_epochs=num_epochs,
                               alpha=10,
                               beta= 0.75)

    return optimizer, lr_scheduler


def _run(
        model:ResnetSimClr,
        
        train_dl: DataLoader,
        val_dl: Optional[DataLoader],

        loss_obj: SimClrLoss,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 

        ckpnt_dir: Optional[Union[str, Path]],

        num_epochs:int,
        val_per_epoch: int,
        device: str,

        ):

    # process the checkpoint directory
    ckpnt_dir = dirf.process_path(ckpnt_dir, 
                                  dir_ok=True, 
                                  file_ok=False)

    # keep two variables: min_train_loss, min_val_loss
    min_train_loss, min_val_loss = float('inf'), float('inf')

    for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):

        epoch_train_loss = train_per_epoch(model=model, 
                        dataloader=train_dl,
                        loss_function=loss_obj,
                        epoch_index=epoch_index,
                        device=device, 
                        log_per_batch=0.1, 
                        optimizer=optimizer,
                        scheduler=lr_scheduler)

        print(f"epoch {epoch_index}: train loss: {epoch_train_loss}")

        if val_dl is None and min_train_loss > epoch_train_loss:
            min_train_loss = epoch_train_loss
            ckpnt_file_name = f'ckpnt_train_loss-{round(min_train_loss, 4)}_epoch-{epoch_index}.pt'

            if len(os.listdir(ckpnt_dir)) != 0:
                # keep only one checkpoint             
                dirf.clear_directory(directory=ckpnt_dir, condition=lambda _: True)

            pu.save_checkpoint(model=model, 
                                optimizer=optimizer, 
                                lr_scheduler=lr_scheduler,
                                path=os.path.join(ckpnt_dir, ckpnt_file_name),
                                train_loss=epoch_train_loss, epoch=epoch_index)

        # compute the performance on validation 
        if val_dl is not None and ((epoch_index + 1) % val_per_epoch == 0):
            epoch_val_loss = validation_per_epoch(model=model, 
            dataloader=val_dl,
            epoch_index=epoch_index + 1, 
            device=device,
            log_per_batch=0.2)
            print(f"epoch {epoch_index}: validation loss: {epoch_val_loss}")

            # save the best checkpoint on validation

            min_val_loss = epoch_val_loss
            ckpnt_file_name = f'ckpnt_val_loss_{round(min_val_loss, 4)}epoch_{epoch_index}.pt'

            if len(os.listdir(ckpnt_dir)) != 0:
                # keep only one checkpoint             
                dirf.clear_directory(directory=ckpnt_dir, condition=lambda _: True)

            pu.save_checkpoint(model=model, 
                                optimizer=optimizer, 
                                lr_scheduler=lr_scheduler,
                                path=os.path.join(ckpnt_dir, ckpnt_file_name),
                                val_loss=epoch_val_loss, 
                                epoch=epoch_index)

    # make sure to return the checkpoint 
    return os.path.join(ckpnt_dir, ckpnt_file_name)

def run_pipeline(model: ResnetSimClr, 

          train_data_folder:Union[str, Path],
          val_data_folder:Optional[Union[str, Path]],

          num_epochs: int, 
          batch_size: int,

          learning_rates: Union[Tuple[float, float], float],  
          temperature: float,

          ckpnt_dir: Union[str, Path],
          val_per_epoch: int = 3,
          seed:int = 69,
          run_name: str = 'sim_clr_run'
          ):    

    wandb.init(project=_WANDB_PROJECT_NAME, 
            name=run_name)

    # get the default device
    device = pu.get_default_device()

    # set the loss object    
    loss_obj = SimClrLoss(temperature=temperature)


    # DATA: 
    train_dl, val_dl = _set_data(train_data_folder=train_data_folder, 
                                val_data_folder=val_data_folder, 
                                batch_size=batch_size, 
                                seed=seed)

    optimizer, lr_scheduler = _set_optimizer(model=model, lrs=learning_rates, num_epochs=num_epochs)

    res = None
    try:
        res = _run(model=model, 
                train_dl=train_dl, 
                val_dl=val_dl, 
                loss_obj=loss_obj, 
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ckpnt_dir=ckpnt_dir,
                num_epochs=num_epochs,
                val_per_epoch=val_per_epoch,
                device=device)

    # this piece of code is taken from the fairSeq (by the Facebook AI research team) as recommmended on the Pytorch forum    
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise e
        
        # at this point, the error is known to be an out of memory error
        pu.cleanup()
        # make sure to close the wandb log
        wandb.finish()
        # stop the program for a 30 seconds before calling the function once again
        sleep(seconds=30)
        batch_size = int(batch_size / 1.2)
        res = run_pipeline(model=model, 
                train_data_folder=train_data_folder, 
                val_data_folder=val_data_folder, 
                num_epochs=num_epochs, 
                batch_size=batch_size,
                learning_rates=learning_rates,
                temperature=temperature,
                seed=seed,
                run_name=run_name)

    wandb.finish()
    return res
