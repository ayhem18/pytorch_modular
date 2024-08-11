"""
This script is written to test (more like debug) the implementation of the nearest neighbors algorithm using the embedding of model
"""

import os,  torch 
import torchvision.transforms as tr 

from typing import Union
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.datasets import FashionMNIST

from mypt.code_utilities import directories_and_files as dirf
from mypt.subroutines.topk_nearest_neighbors import model_embs as me
from mypt.models.simClr.simClrResnet import ResnetSimClr


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main_function(img_class_dir: Union[str, Path],
                  res_dir: Union[str, Path]):

    dirf.process_path(img_class_dir, 
                      dir_ok=True, 
                      file_ok=False, 
                      condition=dirf.image_dataset_directory,
                      error_message=dirf.image_dataset_directory_error)

    img_transform = tr.Compose([tr.Resize(32, 32), tr.ToTensor()])
    # create a standard image classification dataset
    ds = ImageFolder(root=img_class_dir, transform=img_transform)


    # call 
    model = ResnetSimClr(input_shape=(3, 32, 32), output_dim=128, num_fc_layers=1, freeze=True)

    me.topk_nearest_model_ckpnt(results_directory=res_dir, 
                                sample_processing=lambda x: x, 
                                model=model, 
                                model_ckpnt=None,
                                batch_size=32,
                                k_neighbors=3)

if __name__ == '__main__':  
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    dirf.process_path(data_dir, file_ok=False)
    res_dir = os.path.join(SCRIPT_DIR, 'temp_res')
    # this object will download the 
    FashionMNIST(root=data_dir, train=False)

    # let's get this out of the way
    main_function(img_class_dir=data_dir, res_dir=res_dir)
