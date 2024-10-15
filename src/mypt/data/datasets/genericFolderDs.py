"""
This script contains an implementation of a generic Dataset Object that returns all images in a given directory
without imposing a specific structure.
"""
import os

from typing import Union, List, Tuple, Dict
from torch.utils.data import Dataset
from PIL import Image


from mypt.code_utilities import directories_and_files as dirf
from mypt.shortcuts import P


class GenericFolderDS(Dataset):
    @classmethod
    def load_sample(cls, sample_path: P):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")



    def __init__(self, 
                 root: P,
                 transforms: List,
                 image_extensions: Union[List[str], Tuple[str]]=None):
    
        super().__init__()

        # set the `allowed` / `accepted` extensions
        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS
        
        self.image_extensions = image_extensions

        # make sure the dataset can represents an image dataset
        self.root = dirf.process_path(root, 
                                      must_exist=True, # the directory must already exist
                                      dir_ok=True, 
                                      file_ok=False,
                                      # the directory can contain only image files
                                      condition=lambda d: dirf.dir_contains_only_types(d, valid_extensions=self.image_extensions) 
                                      )     
    
        self.transforms = transforms

        # create a path from indices to path samples
        self.idx2path : Dict = {}
        self.data_count :int = None # a variable to save the number of items in the dataset

        # make sure to call the build_index2path method
        self.build_index2path()


    def build_index2path(self):
        counter = 0
        for r, _, files in os.walk(self.root):
            for f in files:
                file_path = os.path.join(r, f)
                self.idx2path[counter] = file_path 
                counter += 1
        # at this point every number maps to a sample path
        self.data_count = counter    

    def __getitem__(self, index):
        # load the image
        sample = self.load_sample(self.idx2path[index])
        # pass it through the passed transforms
        for t in self.transforms:
            sample = t(sample)
        
        return sample

    def __len__(self)->int:
        if self.data_count is None or self.data_count == 0:
            raise ValueError(f"Make sure to set the self.data_count attribute correctly to retrive the size of the dataset in O(1) time !!!")
        return self.data_count
