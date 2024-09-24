"""
This script contains utilities to facilitate working with image classification dataset
"""

import warnings, os

from torchvision.datasets import Imagenette
from typing import Tuple, List

from mypt.shortcuts import P
from mypt.code_utilities import directories_and_files as dirf



def builtin_imagenette2img_cls_ds(ds_folder: P, 
                                  des_folder: P,
                                  allow_download: bool = False,
                                  split_names: List[str] = None,
                                  **ds_kwargs) -> Tuple[P, P]:
    # basically this function takes a folder where the imagenette dataset is already installed
    # and conve    

    ds_folder = dirf.process_path(ds_folder, 
                                  file_ok=False, 
                                  dir_ok=True)
    
    des_folder = dirf.process_path(des_folder, 
                                   file_ok=False, 
                                   dir_ok=True)

    if split_names is None:
        split_names = ['train', 'test']

    if not isinstance(split_names, (List, Tuple)) or len(split_names) != 2:
        raise TypeError(f"the 'split_names' argument is expeced to be 2-element list")


    if 'download' in ds_kwargs:
        raise ValueError(f"The download parameter is set by the function and must not be passed !!.")
    
    if not any(ds_kwargs):
        ds_kwargs = {"split": 'train', 'transform': 'None', 'size': 'full'} 

    try:
        ds = Imagenette(root=ds_folder,     
                download=False,  # setting the download to False requires the dataset to be installed in the 'ds_folder' path 
                **ds_kwargs
        )        

    except RuntimeError as e:
        if 'dataset not found' not in str(e).lower() and allow_download:
            warnings.warn(f"The 'allow_download' is set to True. The dataset will be downloaded in the passed path: {ds_folder}")
        
            ds = Imagenette(root=ds_folder,     
                        download=True,  
                        **ds_kwargs)
        else:
            raise e
        
    # at this point, we know the dataset is installed correctly

    src_train_folder = os.path.join(ds_folder, 'imagenette2', 'train')
    src_val_folder = os.path.join(ds_folder, 'imagenette2', 'val')

    des_train_folder = os.path.join(des_folder, split_names[0])
    des_val_folder = os.path.join(des_folder, split_names[1])

    dirf.copy_directories(src_dir=src_train_folder, des_dir=des_train_folder, copy=True)
    dirf.copy_directories(src_dir=src_val_folder, des_dir=des_val_folder, copy=True)

    # use the dataset object to map the unicode to the class names (take a look at the '_WIND_TO_CLASS' in the IMAGENETTE class source code)
    for folder in [des_train_folder, des_val_folder]:
        for f_name in os.listdir(folder):
            src_path = os.path.join(folder, f_name) 
            des_path = os.path.join(folder, "_".join(ds._WNID_TO_CLASS[f_name][0].split(" ")))
            os.rename(src=src_path, dst=des_path)

    # now a ImageFolder Dataset object can be used as a wrapper for the Imagenette Dataset

    return des_train_folder, des_val_folder


