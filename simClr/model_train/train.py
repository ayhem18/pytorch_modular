from typing import Union, Optional
from pathlib import Path
from tqdm import tqdm

import torchvision.transforms as tr


from mypt.losses.simClrLoss import SimClrLoss
from mypt.data.datasets.parallel_aug_ds import ParallelAugDs
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu

from .train_per_epoch import train_per_epoch, validation_per_epoch
from ..models.resnet.model import ResnetSimClr

# the default data augmentations are selected as per the authors' recommendations
_DEFAULT_DATA_AUGS = [tr.RandomVerticalFlip(p=1), 
                      tr.RandomHorizontalFlip(p=1), 
                      tr.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1),
                      tr.GaussianBlur(kernel_size=(5, 5)),
                      ]

_UNIFORM_DATA_AUGS = [tr.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
                    tr.ToTensor()
                        ]

def train(model: ResnetSimClr, 

          train_data_folder:Union[str, Path],
          val_data_folder:Optional[Union[str, Path]],

          num_epochs: int, 
          batch_size: int,

          temperature: float,
          seed:int = 69,
          ):    

    # get the default device
    device = pu.get_default_device()

    # set the loss object    
    loss_obj = SimClrLoss(temperature=temperature)

    # prepare the dataset object

    train_ds = ParallelAugDs(root=train_data_folder, 
                       output_shape=(224, 224), 
                       augs_per_sample=2, 
                       data_augs=_DEFAULT_DATA_AUGS,
                       uniform_data_augs=_UNIFORM_DATA_AUGS)

    if val_data_folder is not None:
        val_ds = ParallelAugDs(root=val_data_folder, 
                        output_shape=(224, 224), 
                        augs_per_sample=2, 
                        data_augs=_DEFAULT_DATA_AUGS,
                        uniform_data_augs=_UNIFORM_DATA_AUGS)
    else:
        val_ds = None

    # set the dataloaders
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=batch_size,
                                         num_workers=0,
                                         warning=False # the num_workers=0 is set deliberately 
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=train_ds, 
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            )
    else:
        val_dl = None

    for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):
        # call the traing per epoch method
        epoch_train_loss = train_per_epoch(model=model, 
                        dataloader=train_dl,
                        loss_function=loss_obj,
                        epoch_index=epoch_index, # keep the 0-index 
                        device=device, 
                        log_per_batch=0.3)

        print(f"epoch {epoch_index}: train loss: {epoch_train_loss}")

        if val_dl is not None:
            epoch_val_loss = validation_per_epoch(model=model, 
            dataloader=val_dl,
            epoch_index=epoch_index + 1, 
            device=device,
            log_per_batch=0.2)
            print(f"epoch {epoch_index}: validation loss: {epoch_val_loss}")
