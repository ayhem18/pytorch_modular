"""
This script contains the code to prepare the data for training on the object localization task
"""

import os
import xml.etree.ElementTree as ET

from typing import Union, Tuple, List
from pathlib import Path

from mypt.code_utilities import directories_and_files as dirf

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def prepare_data(data_folder: Union[str, Path], ann_folder: Union[str, Path]=None):
    data_folder = dirf.process_path(data_folder, dir_ok=True, file_ok=False, 
                                    condition=lambda d: all([os.path.splitext(f)[-1].lower() in ['.xml'] + dirf.IMAGE_EXTENSIONS for f in os.listdir(d)]), # basically make sure that all files are either images of .xml files
                                    error_message=f'Make sure the directory contains only .xml or image data'
                                    )
    if ann_folder is None:
        ann_folder = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_annotations')
        ann_folder = dirf.process_path(ann_folder, dir_ok=True, file_ok=False)
        
    else:
        ann_folder = dirf.process_path(ann_folder, 
                                       dir_ok=True, 
                                       file_ok=False, 
                                       condition=lambda d: (len(os.listdir(d))) == 0, 
                                       error_message=f"The annotation dir must be empty ")

    # move the .xml files to annotation folder
    dirf.copy_directories(src_dir=data_folder, 
                          des_dir=ann_folder, 
                          copy=False, # setting copy to False will move the files 
                          filter_directories=lambda p: os.path.splitext(p)[-1].lower() == ".xml" #move onle .xml files
                          )
    # everything should be done
    return ann_folder


def split_data_train_val(data_folder: Union[str, Path], val_split:float=0.15):
    train_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_train')
    val_dir = os.path.join(Path(data_folder).parent, f'{os.path.basename(data_folder)}_val')

    dirf.directory_partition(src_dir=data_folder, des_dir=val_dir, copy=False, portion=val_split)
    dirf.copy_directories(src_dir=data_folder, des_dir=train_dir, copy=True) # set copy to False to save disk space


if __name__ == '__main__':
    folder = os.path.join(SCRIPT_DIR, 'plants_ds')    
    # prepare_data(data_folder=folder)
    split_data_train_val(data_folder=folder)
