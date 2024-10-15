"""
This script contains the implementation of a general dataset object designed for Constrastive Learning Parallel Augmentation approaches 
(https://arxiv.org/pdf/2002.05709, https://arxiv.org/abs/2103.03230) for example
"""
import os, random

import torchvision.transforms as tr
from torch.utils.data import Dataset
from typing import Union, List, Tuple
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod

from ....code_utilities import pytorch_utilities as pu


class AbstractParallelAugsDs(Dataset, ABC):
    """
    The parent class of Parallel Augmentation Datasets. The main functionality is implemented in
    in the ._set_augmentations() method: returning 2 tr.Compose each outputting images of the same output shape
    """

    @classmethod
    def load_sample(cls, sample_path: Union[str, Path]):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


    def __init__(self, 
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                uniform_augs_before: List,
                uniform_augs_after:List,
                sampled_data_augs:List,
                seed: int=0):

        # reproducibility is crucial for a consistent evaluation of the model
        pu.seed_everything(seed=seed)
        
        # the output of the resulting data (After augmentation)
        self.output_shape = output_shape

        # make sure each transformation starts by resizing the image to the correct size
        self.sampled_data_augs = sampled_data_augs        
        self.uniform_augs_before = uniform_augs_before 
        self.uniform_augs_after = uniform_augs_after

        self.augs_per_sample = min(augs_per_sample, len(self.sampled_data_augs))


    def _set_augmentations(self) -> Tuple[tr.Compose, tr.Compose]:
        # sample from the passed augmentations
        augs1, augs2 = random.sample(self.sampled_data_augs, self.augs_per_sample), random.sample(self.sampled_data_augs, self.augs_per_sample)

        # convert to a tensor
        augs_before = [tr.ToTensor()] + self.uniform_augs_before

        augs1 = augs_before + augs1  
        augs2 = augs_before + augs2  
    
        # add all the uniform_after augmentations 
    
        augs1.extend(self.uniform_augs_after)
        augs2.extend(self.uniform_augs_after)

        # resize after all transformations:
        augs1.append(tr.Resize(size=self.output_shape))
        augs2.append(tr.Resize(size=self.output_shape))

        # no need to convert to tensors
        augs1, augs2 = tr.Compose(augs1), tr.Compose(augs2)

        return augs1, augs2


    @abstractmethod
    def __getitem__(self, index: int):
        pass
