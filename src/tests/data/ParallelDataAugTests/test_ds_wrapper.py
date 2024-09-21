"""
a short script to verify whether the dataset wrapper works as expected when specifying the number of samples per cls
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from mypt.shortcuts import P
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds_wrapper import Food101Wrapper

OUTPUT_SHAPE = (200, 200)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_ds_wrapper(folder: P,
                     train: bool, 
                     classification_mode:bool,
                    num_samples_per_cls: int):

    ds = Food101Wrapper(root_dir=folder, 
                        output_shape=OUTPUT_SHAPE[1:],
                        augs_per_sample=2, 
                        sampled_data_augs=[],
                        uniform_augs_before=[],
                        uniform_augs_after=[],
                        train=train,
                        samples_per_cls=num_samples_per_cls,
                        classification_mode=classification_mode,
                        )
    

    for i in range(0, num_samples_per_cls * 3, num_samples_per_cls):
        fig = plt.figure() 

        if classification_mode:
            for j in range(5):
                x, y = ds[i + j]            
                x = np.moveaxis(x.numpy(), 0,-1)
                
                fig.add_subplot(1, 5, j + 1) 
                plt.imshow(x)
                plt.title(f"label: {y}")

            plt.show()
        else:
            for j in range(3):
                x, y = ds[i + j]            
                x = np.moveaxis(x.numpy(), 0,-1)
                y = np.moveaxis(y.numpy(), 0,-1)
                
                fig.add_subplot(1, 6, 2 * j + 1) 
                plt.imshow(x)
                fig.add_subplot(1, 6, 2 * (j + 1)) 
                plt.imshow(y)
            
            plt.show()


def check_ds_different_modes():
    # this function is visual inspection rather than an actual test
    t = os.path.join(SCRIPT_DIR, 'data', 'food101', 'train')
    v = os.path.join(SCRIPT_DIR, 'data', 'food101', 'val')

    check_ds_wrapper(t, 
                     train=True, 
                     classification_mode=True,
                     num_samples_per_cls=10)    

    check_ds_wrapper(t, 
                     train=True, 
                     classification_mode=False,
                     num_samples_per_cls=10)    

    check_ds_wrapper(v, 
                     train=False, 
                     classification_mode=True,
                     num_samples_per_cls=10)    

    check_ds_wrapper(v, 
                     train=False, 
                     classification_mode=False,
                     num_samples_per_cls=10)    


if __name__ == '__main__':
    check_ds_different_modes()
