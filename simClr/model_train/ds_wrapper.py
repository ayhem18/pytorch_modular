import torch, numpy as np


import torchvision.transforms as tr

from pathlib import Path 
from typing import Union, List, Tuple, Optional
from bisect import bisect
from tqdm import tqdm

from torchvision.datasets import Food101

from mypt.data.datasets.parallel_augmentation.parallel_aug_abstract import AbstractParallelAugsDs

class Food101Wrapper(AbstractParallelAugsDs):
    def __init__(self, 
                 root_dir: Union[str, Path], 
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_augs_before: List,
                uniform_augs_after: List,
                train:bool=True,
                samples_per_cls: Optional[int] = None,
                classification_mode: bool=False) -> None:
        
        if classification_mode and len(sampled_data_augs) > 0:
            raise ValueError(f"In classification mode, there should be no sampled data augmentations...")

        super().__init__(output_shape=output_shape, 
                         augs_per_sample=augs_per_sample,
                         sampled_data_augs=sampled_data_augs,                         
                         uniform_augs_before=uniform_augs_before,
                         uniform_augs_after=uniform_augs_after)
        
        # add all of the uniform data augmentations
        ds_transform = [tr.ToTensor(), tr.Resize(size=output_shape)] + uniform_augs_before + uniform_augs_after

        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=tr.Compose(ds_transform) if classification_mode else None, 
                         download=True)

        self.samples_per_cls_map = None

        if samples_per_cls is not None:
            self._len = 101 * samples_per_cls
            self.__set_samples_per_cls(samples_per_cls) 
        else:
            self._len = len(self._ds)

        self.classification_mode = classification_mode


    def __set_samples_per_cls(self, samples_per_cls: int):
        # iterate through the dataset
        current_cls = None
        
        last_pointer = 0
        cls_count = 0

        mapping = {0: 0}
        
        for i in tqdm(range(len(self._ds)), desc="iterating through the dataset to set the samples per each class"):
        # for i in tqdm(range(3150), desc="iterating through the dataset to set the samples per each class"):
            _, c = self._ds[i]
            
            if current_cls is None:
                current_cls = c
            
            if current_cls == c:
                cls_count += 1    

                if cls_count == samples_per_cls:
                    last_pointer = last_pointer + cls_count
                    

            else:
                if cls_count <= samples_per_cls:
                    last_pointer = last_pointer + cls_count

                mapping[last_pointer] = i 

                cls_count = 1              
                current_cls = c          

        self.samples_per_cls_map = mapping


    def __get_item_default_(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract the path to the sample (using the map between the index and the sample path !!!)
        if self.samples_per_cls_map is not None:
            stop_points = sorted(list(self.samples_per_cls_map.keys()))

            index = bisect(stop_points, index)
            
            if index == len(stop_points):
                index -= 1

            sp = stop_points[index]

            sample_image:torch.Tensor = self._ds[self.samples_per_cls_map[sp] + index - sp][0]
        else:
            sample_image = self._ds[index][0]   

        augs1, augs2 = self._set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 

        return s1, s2


    def __get_item_cls_(self, index:int) -> Tuple[torch.Tensor, int]:
        # extract the path to the sample (using the map between the index and the sample path !!!)
        if self.samples_per_cls_map is not None:
            stop_points = sorted(list(self.samples_per_cls_map.keys()))

            index = bisect(stop_points, index)
            
            if index == len(stop_points):
                index -= 1

            sp = stop_points[index]

            item = self._ds[self.samples_per_cls_map[sp] + index - sp]
            return item  
        
        item = self._ds[index]
        return item

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, List[int]]
                                               ]:
        if self.classification_mode:
            return self.__get_item_cls_(index)

        return self.__get_item_default_(index)

    def __len__(self) -> int:
        return self._len
