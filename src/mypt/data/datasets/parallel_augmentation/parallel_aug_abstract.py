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

from ....code_utilities import pytorch_utilities as pu

from abc import ABC, abstractmethod

class AbstractParallelAugsDs(Dataset, ABC):
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
                sampled_data_augs:List,
                uniform_data_augs: List,
                seed: int=0):

        # reproducibiliy provides a much better idea about the performance
        pu.seed_everything(seed=seed)
        
        # the output of the resulting data (After augmentation)
        self.output_shape = output_shape

        # make sure each transformation starts by resizing the image to the correct size
        self.sampled_data_augs = sampled_data_augs
        self.uniform_data_augs = uniform_data_augs
        self.augs_per_sample = min(augs_per_sample, len(self.sampled_data_augs))


    def _set_augmentations(self) -> Tuple[tr.Compose, tr.Compose]:
        augs1, augs2 = random.sample(self.sampled_data_augs, self.augs_per_sample), random.sample(self.sampled_data_augs, self.augs_per_sample)

        # convert to a tensor
        augs1.insert(0, tr.ToTensor())
        augs2.insert(0, tr.ToTensor())

        # resize before any specific transformations
        augs1.insert(1, tr.Resize(size=self.output_shape))
        augs2.insert(1, tr.Resize(size=self.output_shape))
    
        # add all the uniform augmentations: applied regardless of the model 
        augs1.extend(self.uniform_data_augs)
        augs2.extend(self.uniform_data_augs)

        # resize after all transformations:
        augs1.append(tr.Resize(size=self.output_shape))
        augs2.append(tr.Resize(size=self.output_shape))

        # no need to convert to tensors,
        augs1, augs2 = tr.Compose(augs1), tr.Compose(augs2)

        return augs1, augs2


    @abstractmethod
    def __getitem__(self, index: int):
        pass



