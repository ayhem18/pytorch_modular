"""
This script is written to test (more like debug) the implementation of the nearest neighbors algorithm using the embedding of model
"""

import os, torch, shutil

import torchvision.transforms as tr 

from typing import Union
from pathlib import Path
from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder
from torchvision.datasets import FashionMNIST

from mypt.code_utilities import directories_and_files as dirf
from mypt.subroutines.neighbors import model_embs as me
from mypt.models.simClr.simClrModel import ResnetSimClr
from mypt.data.datasets.parallel_augmentation.parallel_aug_ds import ParallelAugDs

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def _main_function(ds: Dataset, 
                  img_class_dir,
                  img_transform,
                  res_dir: Union[str, Path],
                  model):


    if img_transform is None:
        img_transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
    # create a standard image classification dataset

    if ds is None:
        img_cls_dir = dirf.process_path(img_class_dir, 
                        dir_ok=True, 
                        file_ok=False, 
                        condition=dirf.image_dataset_directory,
                        error_message=dirf.image_dataset_directory_error)

        ds = ImageFolder(root=img_class_dir, transform=img_transform)

    me.topk_nearest_model_ckpnt(
                                dataset=ds,
                                results_directory=res_dir, 
                                sample_processing=lambda x: x, 
                                model=model, 
                                model_ckpnt=None,
                                batch_size=64,
                                k_neighbors=3)

def test_with_standard_cls_ds(): 
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    dirf.process_path(data_dir, file_ok=False)
    res_dir = os.path.join(SCRIPT_DIR, 'temp_res')
    # this object will download the 
    img_transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

    ds = FashionMNIST(root=data_dir, 
                 train=False, 
                 download=True, 
                 transform=img_transform)

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=32 * 32, out_features=128))

    # let's get this out of the way
    _main_function(ds=ds, 
                  img_transform=None,
                  img_class_dir=data_dir, 
                  res_dir=res_dir, 
                  model=model)
    
    # remove the data and the results directory
    shutil.rmtree(data_dir)
    shutil.rmtree(res_dir)

    
def test_with_clr_ds():
    # MAKE SURE TO HAVE a dataset in the 'data_dir' 
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    dirf.process_path(data_dir, dir_ok=True, file_ok=False, )        

    data_dir = dirf.process_path(data_dir, 
                                dir_ok=True, 
                                file_ok=False, 
                                condition=dirf.image_directory,
                                error_message="The data diretory is expected to contain only image data"
                                )

    ds = ParallelAugDs(root=data_dir, 
                       output_shape=(224, 224), 
                       augs_per_sample=1, 
                       sampled_data_augs=[tr.RandomVerticalFlip(p=1)],
                       uniform_data_augs=[])

    model = ResnetSimClr(input_shape=(3, 224, 224), output_dim=64, num_fc_layers=2, freeze=True)

    res_dir = os.path.join(SCRIPT_DIR, 'temp_res')

    _main_function(ds=ds, 
                  img_transform=None,
                  img_class_dir=data_dir, 
                  res_dir = res_dir,
                  model=model)

    shutil.rmtree(data_dir)
    shutil.rmtree(res_dir)


if __name__ == '__main__':  
    test_with_clr_ds()
