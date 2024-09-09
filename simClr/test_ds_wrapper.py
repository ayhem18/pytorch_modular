"""
a short script to verify whether the dataset wrapper works as expected when specifying the number of samples per cls
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from mypt.shortcuts import P
from model_train.ds_wrapper import Food101Wrapper
from model_train.training import OUTPUT_SHAPE, _UNIFORM_DATA_AUGS


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_ds_wrapper(folder: P,
                     train: bool, 
                     classification_mode:bool,
                    num_samples_per_cls: int):
    # define the dataset
    ds = Food101Wrapper(root_dir=folder, 
                        output_shape=OUTPUT_SHAPE[1:],
                        augs_per_sample=2, 
                        sampled_data_augs=[],
                        uniform_augs_before=[],
                        uniform_augs_after=_UNIFORM_DATA_AUGS,
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

if __name__ == '__main__':
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


