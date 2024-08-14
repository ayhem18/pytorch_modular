"""
This script
"""
import os, numpy as np, torch
import matplotlib.pyplot as plt
import torchvision.transforms as tr

from PIL import Image
from pathlib import Path

from mypt.data.datasets.parallel_aug_ds import ParallelAugDs
from ds_wrapper import CALTECH101Wrapper

from mypt.code_utilities import pytorch_utilities as pu
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader
from mypt.losses.simClrLoss import SimClrLoss

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):    
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


_DEFAULT_DATA_AUGS = [ 
                      tr.RandomHorizontalFlip(p=1), 
                      tr.RandomRotation(degrees=10),
					#   tr.RandomErasing(p=1, scale=(0.05, 0.15)),
					  tr.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),
                      tr.GaussianBlur(kernel_size=(5, 5)),
                      ]

_UNIFORM_DATA_AUGS = [#tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]


def check_data_aug(data_folder):

    train_ds = CALTECH101Wrapper(root_dir=data_folder, 
                                output_shape=(96, 96),
                                augs_per_sample=2, 
                                sampled_data_augs=_DEFAULT_DATA_AUGS,
                                uniform_data_augs=_UNIFORM_DATA_AUGS)

    # create a dataloader
    dl = initialize_train_dataloader(train_ds, seed=0, batch_size=10, num_workers=0)

    for i, (x1, x2) in enumerate(dl):
        x = torch.cat([x1, x2])

        n = len(x1)

        fig = plt.figure() 

        for j in range(n):
            p1, p2 = x[j].squeeze(), x[(j + n) % (2 * n)].squeeze()
            p1, p2 = np.moveaxis(p1.numpy(), 0,-1), np.moveaxis(p2.numpy(), 0, -1)

            fig.add_subplot(1, 2, 1) 
            plt.imshow(p1) 
            plt.title("augmented image 1") 

            fig.add_subplot(1, 2, 2) 
            plt.imshow(p2) 
            plt.title("augmented image 2") 

            plt.show()

    # for i in range(5):
    #     # im = train_ds.load_sample(train_ds.idx2path[i])
    #     x1, x2 = train_ds[i]

    #     # im = np.array(im)
    #     x1, x2 = x1.numpy(), x2.numpy()

    #     # transpose the image 
    #     x1, x2 = np.moveaxis(x1, 0,-1), np.moveaxis(x2, 0, -1)
    #     fig = plt.figure() 
    #     # fig.add_subplot(1, 2, 1) 
    #     # plt.imshow(im) 
    #     # plt.title("original image") 

    #     fig.add_subplot(1, 2, 1) 
    #     plt.imshow(x1) 
    #     plt.title("augmented image 1") 

    #     fig.add_subplot(1, 2, 2) 
    #     plt.imshow(x2) 
    #     plt.title("augmented image 2") 

    #     plt.show()



if __name__ == '__main__':
    train_data_folder = os.path.join(DATA_FOLDER, 'caltech101', 'train')
    check_data_aug(train_data_folder)
