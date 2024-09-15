"""
This script contains few tests to see whether the current implementation can learn the extremely simple MNIST dataset
"""

import torch, os, shutil

import torchvision.transforms as tr

from tqdm import tqdm
from torch.optim.adam import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import mypt.code_utilities.pytorch_utilities as pu

from mypt.shortcuts import P
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from mypt.linearBlocks.fc_block_components  import ResidualLinearBlock
from mypt.linearBlocks.fully_connected_blocks  import GenericFCBlock


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _set_data(data_folder: P):
    ds = MNIST(root=data_folder, 
               train=False, 
               transform=tr.ToTensor(), 
               download=True)
    
    dl = initialize_train_dataloader(dataset_object=ds, 
                                     seed=0, 
                                     batch_size=32, 
                                     num_workers=2, 
                                     drop_last=True) 
    
    return dl

def _run_single_classifier(rlb: ResidualLinearBlock, 
                           dl: DataLoader,
                           lr:float,
                           num_epochs: int = 20):

    pu.seed_everything(0)

    device = pu.get_default_device() 
    
    rlb = rlb.to(device)
    rlb.train()

    optimizer = Adam(rlb.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    losses = []
    

    new_shape = (32, 28 * 28)
    for _ in tqdm(range(num_epochs), desc='training the classifier'):
        epoch_loss = 0 
    
        for x, y in dl: 

            x, y =  x.to(device).reshape(new_shape), y.to(device)

            optimizer.zero_grad()

            out = rlb.forward(x)

            batch_loss = loss_function.forward(out, y)
            epoch_loss += batch_loss.item()

            batch_loss.backward()

            optimizer.step()

        losses.append(epoch_loss / len(dl))

    print("#" * 20)
    print(losses[-10:]) # only the last 10 losses
    print("#" * 20)

    return losses        

def test_residual_block():
    data_folder = os.path.join(SCRIPT_DIR, 'data')

    n = 28 * 28

    print("TRAINING RESIDUAL BLOCKS !!")

    for i in range(4, 7):
        # get the dataset
        dl = _set_data(data_folder)
        # define the classifier
        classifier = ResidualLinearBlock(output=10, in_features=n, num_layers=i, units= [n] + [2 ** (10 - j) for j in range(i - 1)] + [10])

        # train and see how it goes
        losses = _run_single_classifier(rlb=classifier, dl=dl, lr=10 ** -2)

    # run a similar procedure for generic linear blocks and see if the residual fitting improves the performance

    print("TRAINING GENERIC LINEAR BLOCKS", end='\n\n')

    for i in range(4, 7):
        # get the dataset
        dl = _set_data(data_folder)
        # define the classifier
        classifier = GenericFCBlock(output=10, in_features=n, num_layers=i, units= [n] + [2 ** (10 - j) for j in range(i - 1)] + [10])
        # train and see how it goes
        losses = _run_single_classifier(rlb=classifier, dl=dl, lr=10 ** -2)

    # remove the data 
    shutil.rmtree(data_folder)


if __name__ == '__main__':
    test_residual_block()
