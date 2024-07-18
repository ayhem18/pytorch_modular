import torch, random, wandb

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Union

from .models.simClrModel import SimClrModel
from mypt.code_utilities import pytorch_utilities as pu
from mypt.losses.simClrLoss import SimClrLoss

def train_per_batch(model: SimClrModel,
                    x1_batch: torch.Tensor,
                    x2_batch: torch.Tensor,
                    loss_function: SimClrLoss,
                    optimizer: torch.optim.Optimizer,
                    ) -> float:
    
    device = pu.get_module_device(model)
    # make sure to stack the two batches into a single batch 
    x = torch.cat([x1_batch, x2_batch]).to(device)
    # set the optimizer's gradients to zero
    optimizer.zero_grad()
    
    # forward pass through the model
    _, g_x = model.forward(x)

    batch_loss_obj = loss_function.forward(g_x)

    batch_loss = batch_loss_obj.item()

    # perform the backpropagation
    batch_loss_obj.backward()

    # perform update the weights
    optimizer.step()

    return batch_loss

def train_per_epoch(model: nn.Module, 
                dataloader: DataLoader,
                loss_function: nn.Module,
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler.LRScheduler, 
                epoch_index: int,
                device: str,
                log_per_batch: Union[int, float],
                ) -> Tuple[float, float]:

    if isinstance(log_per_batch, float):
        num_batches = len(dataloader)
        log_per_batch = int(num_batches * log_per_batch)

    # set the model to the train mode
    model = model.to(device=device)
    model.train()


    # define a function to save the average loss per epoch
    epoch_train_loss = 0

    for batch_index, (x1, x2) in tqdm(enumerate(dataloader), desc=f'training batch at epoch {epoch_index }'): 
        batch_train_loss = train_per_batch(model=model, 
                                                x1_batch=x1, 
                                                x2_batch=x2, 
                                                loss_function=loss_function,
                                                optimizer=optimizer)
        
        # log the batch loss depending on the batch index
        if batch_index % log_per_batch:
            wandb.log({"epoch": epoch_index, "train_loss": batch_train_loss})


        epoch_train_loss += epoch_train_loss

    # make sure to call the scheduler to update the learning rate
    scheduler.step()

    # make sure to average the metric
    epoch_train_loss = epoch_train_loss / len(dataloader)

    # log the metrics
    wandb.log({"epoch": epoch_index, 
               "train_loss": epoch_train_loss})

    return epoch_train_loss

def validation_per_batch(model: nn.Module, 
                        x1_batch: torch.Tensor,
                        x2_batch: torch.Tensor,
                        loss_function: nn.Module,
                     ):

    device = pu.get_module_device(model)
    # make sure to stack the two batches into a single batch 
    x = torch.cat([x1_batch, x2_batch]).to(device)

    with torch.no_grad():
        # forward pass through the model
        _, g_x = model.forward(x)

        batch_loss_obj = loss_function.forward(g_x)

        batch_loss = batch_loss_obj.item()

    return batch_loss

def validation_per_epoch(model: nn.Module,
                         dataloader: DataLoader,
                         epoch_index: int,
                         device: str, 
                        log_per_batch: Union[float, int]) -> Tuple[float, float]:

    if isinstance(log_per_batch, float):
        num_batches = len(dataloader)
        log_per_batch = int(num_batches * log_per_batch)

    epoch_val_loss = 0
    loss_function = nn.CrossEntropyLoss()        

    model = model.to(device=device)
    model.eval()
    for batch_index, (x, y) in tqdm(enumerate(dataloader), desc=f'validation batch for epoch {epoch_index}'):
        batch_loss = validation_per_batch(model=model, 
                                                          batch_x=x, 
                                                          batch_y=y, 
                                                          loss_function=loss_function)

        if batch_index % log_per_batch == 0:
            wandb.log({"epoch": epoch_index, "val_loss": batch_loss})

        # add the batch loss to the epoch loss         
        epoch_val_loss += batch_loss

    wandb.log({"epoch": epoch_index, "val_loss": epoch_val_loss / len(dataloader)})
