import wandb, torch, os, math

import torchvision.transforms as tr

from time import sleep
from typing import Union, Optional, Tuple, List
from pathlib import Path

from tqdm import tqdm
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader


from mypt.losses.simClrLoss import SimClrLoss
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.schedulers.annealing_lr import AnnealingLR
from mypt.models.simClr.simClrModel import SimClrModel

from .train_per_epoch import train_per_epoch, validation_per_epoch
from .ds_wrapper import Food101Wrapper

from flash.core.optimizers import LARS

_WANDB_PROJECT_NAME = "SimClr"
PREFERABLE_BATCH_SIZE = 512
OUTPUT_SHAPE = (3, 200, 200)


_DEFAULT_DATA_AUGS = [
    tr.RandomHorizontalFlip(p=0.5), 
    tr.RandomResizedCrop(OUTPUT_SHAPE[1:], scale=(0.4, 1)),
    
    tr.RandomErasing(p=0.5, scale=(0.05, 0.15)),
    tr.RandomGrayscale(p=0.5),
    tr.GaussianBlur(kernel_size=(5, 5)),
    tr.ColorJitter(brightness=0.5, contrast=0.5)
]

_UNIFORM_DATA_AUGS = [] # [tr.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])] : one of the augmentations used with the original Resnet models 
                    

def _set_data(train_data_folder: Union[str, Path],
            val_data_folder: Optional[Union[str, Path]],
            batch_size:int,
            output_shape: Tuple[int, int],
            num_train_samples_per_cls:Optional[int],
            num_val_samples_per_cls:Optional[int],    
            seed:int=69) -> Tuple[DataLoader, Optional[DataLoader]]:
    
    train_data_folder = dirf.process_path(train_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

    train_ds = Food101Wrapper(root_dir=train_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=True,
                                samples_per_cls=num_train_samples_per_cls
                                )

    if val_data_folder is not None:
        val_data_folder = dirf.process_path(val_data_folder, 
                                          dir_ok=True, 
                                          file_ok=False, 
                                          )

        val_ds = Food101Wrapper(root_dir=val_data_folder, 
                                output_shape=output_shape,
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_augs_before=[],
                                uniform_augs_after=_UNIFORM_DATA_AUGS,
                                train=False,
                                samples_per_cls=num_train_samples_per_cls
                                )

    else:
        val_ds = None

    ## data loaders
    train_dl = initialize_train_dataloader(dataset_object=train_ds, 
                                         seed=seed,
                                         batch_size=batch_size,
                                         num_workers=2,
                                         warning=False # the num_workers=0 is deliberately set to 0  
                                         )

    if val_ds is not None:
        val_dl = initialize_val_dataloader(dataset_object=train_ds, 
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_workers=2,
                                            )
    else:
        val_dl = None

    return train_dl, val_dl


def _set_optimizer(model: SimClrModel, 
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

def _set_optimizer(model: SimClrModel,
                   learning_rate: float) -> LARS:

    return LARS(params=model.parameters(), lr=learning_rate, weight_decay=10 ** -6,)

def _run(
        model:SimClrModel,
        
        train_dl: DataLoader,
        val_dl: Optional[DataLoader],

        loss_obj: SimClrLoss,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
        
        accumulate_grad: int,

        ckpnt_dir: Optional[Union[str, Path]],

        num_epochs:int,
        val_per_epoch: int,
        device: str,

        use_wandb:bool=True,
        batch_stats:bool=False
        ):

    # process the checkpoint directory
    ckpnt_dir = dirf.process_path(ckpnt_dir, 
                                  dir_ok=True, 
                                  file_ok=False)

    # keep two variables: min_train_loss, min_val_loss
    min_train_loss, min_val_loss = float('inf'), float('inf')

    for epoch_index in tqdm(range(num_epochs), desc=f'training the model'):
        
        # train the model for one epoch
        epoch_train_metrics = train_per_epoch(model=model, 
                        dataloader=train_dl,
                        loss_function=loss_obj,
                        epoch_index=epoch_index,
                        device=device, 
                        log_per_batch=0.1, 
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        use_wandb=use_wandb,
                        batch_stats=batch_stats,
                        accumulate_grads=accumulate_grad)

        ################### Logging ###################
        epoch_train_log = epoch_train_metrics.copy()
        

        # the dict object cls not allow altering the keys while iterating through the object
        items = list(epoch_train_log.items())

        for k, val in items:
            epoch_train_log.pop(k)
            epoch_train_log[f'train_{k}'] = val

        print("#" * 25)
        print(epoch_train_log)

        # extract the epoch_train_loss
        epoch_train_loss = epoch_train_metrics["loss"]

        ######################## Train Loss Checkpointing ########################        

        if val_dl is None: # meaning we are saving checkpoints according to the trainloss
            if min_train_loss < epoch_train_loss:
                continue 

            # at this point the epoch training loss is the minimum so far
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

            continue

        
        # make sure to update the minimum training loss
        min_train_loss = min(min_train_loss, epoch_train_loss)

        # at this point we know that val_dl is not None and hence the validation loss is the criteria to save checkpoints
        # run the model on the validation set every "val_per_epoch" and the last epoch
        if not ((epoch_index + 1) % val_per_epoch == 0 or epoch_index == num_epochs - 1):
            continue

        epoch_val_metrics = validation_per_epoch(model=model, 
                                        dataloader=val_dl,
                                        loss_function=loss_obj,
                                        epoch_index=epoch_index + 1, 
                                        device=device,
                                        log_per_batch=0.2,
                                        use_wandb=use_wandb,
                                        batch_stats=batch_stats
                                        )

        # add the 'val' to each key in the 'epoch_val_metrics' dictionary
        epoch_val_log = epoch_val_metrics.copy()
        items = list(epoch_val_log.items())
        
        for k, val in items:
            epoch_val_log.pop(k)
            epoch_val_log[f'val_{k}'] = val
        
        print("#" * 25)
        print(epoch_val_log)

        # extract the validation loss
        epoch_val_loss = epoch_val_metrics["loss"]

        if epoch_val_loss > min_val_loss:
            continue


        # at this point epoch_val_loss is the minimal validation loss achieved so far
        min_val_loss = epoch_val_loss
        ckpnt_file_name = f'ckpnt_val_loss_{round(min_val_loss, 4)}_epoch_{epoch_index}.pt'

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
    final_ckpnt = os.path.join(ckpnt_dir, ckpnt_file_name) 

    if val_dl is not None:
        return min_train_loss, min_val_loss, final_ckpnt 

    return min_train_loss, final_ckpnt


def run_pipeline(model: SimClrModel, 
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        output_shape:Tuple[int, int], 

        num_epochs: int, 
        batch_size: int,

        # learning_rates: Union[Tuple[float, float], float],
        temperature: float,

        ckpnt_dir: Union[str, Path],
        val_per_epoch: int = 3,
        seed:int = 69,
        run_name: str = 'sim_clr_run',
        use_wandb: bool = True,
        batch_stats:bool=False,
        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ):    

    if use_wandb:
        # this argument was added to avoid initializing wandb twice during the tuning process
        wandb.init(project=_WANDB_PROJECT_NAME, 
                name=run_name)

    # get the default device
    device = pu.get_default_device()

    
    # set the loss object    
    loss_obj = SimClrLoss(temperature=temperature)

    # DATA: 
    train_dl, val_dl = _set_data(train_data_folder=train_data_folder,
                                val_data_folder=val_data_folder, 
                                output_shape=output_shape,
                                batch_size=batch_size, 
                                num_train_samples_per_cls=num_train_samples_per_cls,
                                num_val_samples_per_cls=num_val_samples_per_cls,
                                seed=seed)

    # # dealing with gradient accumulation
    # accumulate_grads_factor = PREFERABLE_BATCH_SIZE / batch_size 

    # # scale the learning rate by the gradient accumulation factor
    # if isinstance(learning_rates, (Tuple, List)):
    #     learning_rates = [lr / accumulate_grads_factor for lr in learning_rates]
    # else:
    #     learning_rates /= accumulate_grads_factor
    
    # # make sure to convert it to an integer
    # accumulate_grads_factor = int(math.ceil(accumulate_grads_factor))

    # # optimizer = _set_optimizer(model=model, learning_rate=initial_lr)
    # optimizer, lr_scheduler = _set_optimizer(model=model, lrs=learning_rates, num_epochs=num_epochs)

    lr = int(0.3 * (batch_size / 256)) # according to the paper formula

    optimizer = _set_optimizer(model=model, 
                               learning_rate=lr  
                               )

    res = None
    try:
        res = _run(model=model, 
                train_dl=train_dl, 
                val_dl=val_dl, 

                loss_obj=loss_obj, 
                optimizer=optimizer,
                lr_scheduler=None,
                
                accumulate_grad=1, # keep it simple, no gradient accumulation for now

                ckpnt_dir=ckpnt_dir,
                num_epochs=num_epochs,
                val_per_epoch=val_per_epoch,
                device=device,
                use_wandb=use_wandb, 
                batch_stats=batch_stats
                )

    # this piece of code is taken from the fairSeq repo (by the Facebook AI research team) as recommmended on the Pytorch forum:
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            print(e)
            raise e
        
        # at this point, the error is known to be an out of memory error
        pu.cleanup()
        # make sure to close the wandb log
        wandb.finish()
        # stop the program for 30 seconds for the cleaning to take effect
        sleep(30)

        batch_size = int(batch_size / 1.2)
        res = run_pipeline(model=model, 
                                     
            train_data_folder=train_data_folder,

            val_data_folder=val_data_folder,

            output_shape=output_shape, 

            num_epochs=num_epochs, 
            batch_size=batch_size,

            # initial_lr=initial_lr,  
            # learning_rates=learning_rates,
            temperature=temperature,

            ckpnt_dir=ckpnt_dir,
            val_per_epoch=val_per_epoch,
            seed=seed,
            run_name=run_name, 
            use_wandb=use_wandb, 
            batch_stats=batch_stats           
            )

    
    wandb.finish()
    return res
