"""
This script contains the code to tune the SimClr models on both the 
"""

import os

from pathlib import Path
from typing import Tuple

from mypt.shortcuts import P
from mypt.data.datasets.image_classification import utils as icu
from mypt.code_utilities import directories_and_files as dirf

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR
while 'data' not in os.listdir(current):
    current = Path(current).parent

DATA_FOLDER = os.path.join(current, 'data')


VAL_SPLIT = 0.15

def tuning_data_preparation(imagenette_ds_folder: P,
                            des_data_folder: P) -> Tuple[P, P, P]:
    """
    This function assumes the dataset was downloaded using the builtin Pytorch dataset. 
    The data will be moved to a different direction
    """
    train_split, test_split = icu.builtin_imagenette2img_cls_ds(ds_folder=imagenette_ds_folder, 
                                            des_folder=des_data_folder,
                                            split_names=None, # use the default names
                                            )
    
    val_split = os.path.join(Path(train_split).parent, 'val') 

    # split the train data further into training and validation (the validation split will be used for tuning purposes) 
    dirf.classification_ds_partition(directory_with_classes=train_split, 
                                     destination_directory=val_split, 
                                     portion=VAL_SPLIT,
                                     copy=False)

    return train_split, val_split, test_split
    # the next step is to simply wrap the folders with instances of the DataFolder class
    

if __name__ == '__main__':  
    imagenette_ds_folder = os.path.join(DATA_FOLDER, 'imagenette', 'train')
    imagenette_tune = os.path.join(DATA_FOLDER, 'imagenette_tune')

    train_folder, val_folder, test_folder = tuning_data_preparation(imagenette_ds_folder=imagenette_ds_folder, 
                                                                    des_data_folder=imagenette_tune)
