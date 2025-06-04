import os

from torchvision import datasets
from typing import List, Optional
from torch.utils.data import Dataset

from mypt.shortcuts import P
from mypt.data.datasets.mixins.cls_ds_wrapper import ClassificationDsWrapperMixin


class MiniMnistWrapper(ClassificationDsWrapperMixin, Dataset):
    def __init__(self, 
                 root_dir: P,
                 train: bool,
                 augmentations: Optional[List] = None,
                 samples_per_cls: Optional[int] = None):
        super().__init__(root_dir, train, augmentations)
        
        self._ds = datasets.MNIST(root=root_dir, 
                                 train=train, 
                                 download=not os.path.exists(root_dir),
                                 transform=None
                                 )

        self._samples_per_cls_map = self._set_samples_per_cls(samples_per_cls)
        
    def __getitem__(self, index: int):
        final_index = self._find_final_index(index)
        return self._ds[final_index]
    
    def __len__(self):
        return len(self._ds)
        
        
    