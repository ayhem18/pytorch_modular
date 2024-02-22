"""
This script mainly contains the code to train an initial model for the classification task 
"""

import os, torch
import mypt.utilities.directories_and_files as dirf
import wandb


from functools import partial

from torch import nn
from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from mypt.backbones.resnetFeatureExtractor import ResNetFeatureExtractor
from mypt.classification.classification_head import ExponentialClassifier
from mypt.utilities.pytorch_utilities import set_everything, set_worker_seed, get_default_device
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader, initialize_val_dataloader

# set the project name
WANDB_PROJECT_NAME = 'PLANTS_CLASSIFICAITON'

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, 'data')

def set_data():
    train_dir = os.path.join(DATA_FOLDER, 'train')
    val_dir = os.path.join(DATA_FOLDER, 'val')
    if not os.path.exists(val_dir):
        # create the val directory
        os.makedirs(val_dir)
        dirf.dataset_portion(directory_with_classes=train_dir, 
                             destination_directory=val_dir, 
                             portion=0.15,
                             copy=False)
    return train_dir, val_dir


def train_fast(seed: int = 69):
    # let's set 
    set_everything(seed=seed)

    device = get_default_device() 
    dir_train, dir_val = set_data()

    # let's set some model
    fe = ResNetFeatureExtractor(num_layers=-1, 
                                add_fc=False, 
                                freeze_layers=True, 
                                freeze=3) # do not freeze the last layer
    
    # let's say we have 2 layer for classification
    ch = ExponentialClassifier(in_features=2048, num_classes=12, num_layers=2)

    model = nn.Sequential(fe, nn.Flatten(), ch)

    # create the dataset and dataloaders 
    train_ds = ImageFolder(root=dir_train, transform=fe.transform) 
    val_ds = ImageFolder(root=dir_val, transform=fe.transform) 

    dl_train = initialize_train_dataloader(dataset_object=train_ds, 
                                   seed=seed, 
                                   batch_size=128,
                                   num_workers=3)
    
    dl_val = initialize_val_dataloader(dataset_object=val_ds, 
                                       seed=seed, 
                                       batch_size=120, 
                                       num_workers=0)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)  
    # set the model to 'device'
    model = model.to(device=device)

    # before training let's set everything correctly
    wandb.init(project=WANDB_PROJECT_NAME)

    # time to track the gradients of the model
    wandb.watch(models=model, 
                log_freq=20)

    for epoch_index in tqdm(range(5), desc='training epochs'):
        epoch_train_loss = 0 
        epoch_train_accuracy = 0
        
        # set the model to 'train' every time
        model.train()
        for batch_index, (x, y) in tqdm(enumerate(dl_train), desc=f'training batch at epoch {epoch_index + 1}'): 
            # set the optimizer
            optimizer.zero_grad()
            
            # the first idea is to set the model 
            x, y = x.to(device), y.to(device)
            # forward pass
            y_logits = model.forward(x)
            loss_object = loss_function.forward(input=y_logits, target=y)

            # metrics: 
            epoch_train_loss += loss_object.item()
            batch_accuracy = torch.mean((torch.argmax(y_logits, dim=-1) == y).to(torch.float32))
            epoch_train_accuracy += batch_accuracy

            # backward
            loss_object.backward()
            optimizer.step()
            # scheduler step goes here

            # log the loss of the batch 
            if batch_index % 20 == 0:
                wandb.log({"epoch": epoch_index, "train_loss": loss_object.item()}) 
                wandb.log({"epoch": epoch_index, "train_accuracy": batch_accuracy}) 

        # log the results for the epoch overall
        wandb.log({"epoch": epoch_index, "train_loss": epoch_train_loss / len(dl_train)})
        wandb.log({"epoch": epoch_index, "train_accuracy": epoch_train_accuracy / len(dl_train)})
        
        # report results: 
        # print("#" * 10)1
        # print(f"Train loss for epoch: {epoch_index + 1}, {epoch_train_loss / len(dl_train)}")
        # print(f"Train accuracy for epoch: {epoch_index + 1}, {epoch_train_accuracy / len(dl_train)}")
        # print("#" * 10)

        epoch_val_loss = 0
        epoch_val_accuracy = 0

        # eval: model
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dl_val, desc=f'validation batch for epoch {epoch_index + 1}'):
                # the first idea is to set the model 
                x, y = x.to(device), y.to(device)
                # forward pass
                y_logits = model.forward(x)
                loss_object = loss_function.forward(input=y_logits, target=y)

                # metrics: 
                epoch_val_loss += loss_object.item()
                epoch_val_accuracy += torch.mean((torch.argmax(y_logits, dim=-1) == y).to(torch.float32))

        wandb.log({"epoch": epoch_index, "val_loss": epoch_val_loss / len(dl_val)})
        wandb.log({"epoch": epoch_index, "val_accuracy": epoch_val_accuracy / len(dl_val)})

        # print("#" * 10)
        # print(f"Val loss for epoch: {epoch_index + 1}, {epoch_val_loss / len(dl_val)}")
        # print(f"Val accuracy for epoch: {epoch_index + 1}, {epoch_val_accuracy / len(dl_val)}")
        # print("#" * 10)
        

if __name__ == '__main__':
    train_fast()
