import wandb, torch

import torchvision.transforms as tr

from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm
from torch.optim.sgd import SGD


from mypt.losses.simClrLoss import SimClrLoss
from mypt.data.datasets.parallel_aug_ds import ParallelAugDs
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu
from mypt.schedulers.annealing_lr import AnnealingLR

from .train_per_epoch import train_per_epoch, validation_per_epoch
from .models.resnet.model import ResnetSimClr

# the default data augmentations are selected as per the authors' recommendations
_DEFAULT_DATA_AUGS = [tr.RandomVerticalFlip(p=1), 
                      tr.RandomHorizontalFlip(p=1), 
                    #   tr.ColorJitter(brightness=0.05, contrast=0.05),
                      tr.GaussianBlur(kernel_size=(5, 5)),
                      ]

_UNIFORM_DATA_AUGS = [tr.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])]

_WANDB_PROJECT_NAME = "SimClr"

def train(model: ResnetSimClr, 

          train_data_folder:Union[str, Path],
          val_data_folder:Optional[Union[str, Path]],

          num_epochs: int, 
          batch_size: int,

          temperature: float,
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
    ## Dataset objects
    train_ds = ParallelAugDs(root=train_data_folder, 
                       output_shape=(224, 224), 
                       augs_per_sample=2, 
                       sampled_data_augs=_DEFAULT_DATA_AUGS,
                       uniform_data_augs=_UNIFORM_DATA_AUGS)

    if val_data_folder is not None:
        val_ds = ParallelAugDs(root=val_data_folder, 
                        output_shape=(224, 224), 
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


    lr1, lr2 = 0.001, 0.01

    # set the optimizer
    optimizer = SGD(params=[{"params": model.fe.parameters(), "lr": lr1}, # using different learning rates with different components of the model 
                            {"params": model.flatten_layer.parameters(), "lr": lr1},
                            {"params": model.ph.parameters(), "lr": lr2}
                            ])

    lr_scheduler = AnnealingLR(optimizer=optimizer, num_epochs=num_epochs,alpha=10,beta= 0.75)
    try:
        for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):
            epoch_train_loss = train_per_epoch(model=model, 
                            dataloader=train_dl,
                            loss_function=loss_obj,
                            epoch_index=epoch_index,
                            device=device, 
                            log_per_batch=0.3, 
                            optimizer=optimizer,
                            scheduler=lr_scheduler)
        
            print(f"epoch {epoch_index}: train loss: {epoch_train_loss}")

            if val_dl is not None:
                epoch_val_loss = validation_per_epoch(model=model, 
                dataloader=val_dl,
                epoch_index=epoch_index + 1, 
                device=device,
                log_per_batch=0.2)
                print(f"epoch {epoch_index}: validation loss: {epoch_val_loss}")
    

    # this piece of code is taken from the fairSeq (by the Facebook AI research team) as recommmended on the Pytorch forum
    except RuntimeError as e:
        
        if 'out of memory' not in str(e):
            raise e
        
        # at this point, the error is known to be an out of memory error
        pu.cleanup()
        # make sure to close the wandb log
        wandb.finish()
        batch_size = int(batch_size / 1.2)
        train(model=model, 
            train_data_folder=train_data_folder, 
            val_data_folder=val_data_folder, 
            num_epochs=num_epochs, 
            batch_size=batch_size,
            temperature=temperature,
            seed=seed,
            run_name=run_name)

    wandb.finish()
