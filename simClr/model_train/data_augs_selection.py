"""
This script
"""
import os, numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tr

from PIL import Image
from pathlib import Path

from mypt.data.datasets.parallel_aug_ds import ParallelAugDs
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

current = SCRIPT_DIR
while 'data' not in os.listdir(current):    
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


_DEFAULT_DATA_AUGS = [tr.RandomVerticalFlip(p=1), 
                      tr.RandomHorizontalFlip(p=1), 
                    #   tr.ColorJitter(brightness=0.05, contrast=0.05),
                      tr.GaussianBlur(kernel_size=(5, 5)),
                      ]

_UNIFORM_DATA_AUGS = [tr.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])]


def check_data_aug(data_folder,
              sampled_data_augs):
    train_ds = ParallelAugDs(root=data_folder, 
                      output_shape=(224, 224), 
                      augs_per_sample=2, 
                      sampled_data_augs=sampled_data_augs,
                      uniform_data_augs=_UNIFORM_DATA_AUGS)

    for i in range(5):
        im = train_ds.load_sample(train_ds.idx2path[i])
        x1, x2 = train_ds[i]

        im = np.array(im)
        x1, x2 = x1.numpy(), x2.numpy()

        # transpose the image 
        x1, x2 = np.moveaxis(x1, 0,-1), np.moveaxis(x2, 0, -1)
        fig = plt.figure() 
        fig.add_subplot(1, 3, 1) 
        plt.imshow(im) 
        plt.title("original image") 

        fig.add_subplot(1, 3, 2) 
        plt.imshow(x1) 
        plt.title("augmented image 1") 

        fig.add_subplot(1, 3, 3) 
        plt.imshow(x2) 
        plt.title("augmented image 2") 

        plt.show()



if __name__ == '__main__':
    train_data_folder = os.path.join(DATA_FOLDER, 'plant_seedlings', 'unlabeled_data')
    check_data_aug(train_data_folder, sampled_data_augs=_DEFAULT_DATA_AUGS)
