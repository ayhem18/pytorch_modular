import random, torch


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
                uniform_data_augs: List,
                train:bool=True,
                samples_per_cls: Optional[int] = None) -> None:
        
        super().__init__(output_shape=output_shape, 
                         augs_per_sample=augs_per_sample,
                         sampled_data_augs=sampled_data_augs,
                         uniform_data_augs=uniform_data_augs)
        
        self._ds = Food101(root=root_dir,     
                         split='train' if train else 'test',
                         transform=tr.ToTensor(), 
                         download=True)

        self.samples_per_cls_map = None

        if samples_per_cls is not None:
            self._len = 101 * samples_per_cls
            self.__set_samples_per_cls(samples_per_cls) 
        else:
            self._len = len(self._ds)

    def __set_samples_per_cls(self, samples_per_cls: int):
        # iterate through the dataset
        current_cls = None
        
        last_pointer = 0
        cls_count = 0

        mapping = {0: 0}
        
        # for i in tqdm(range(len(self._ds)), desc="iterating through the dataset to set the samples per each class"):
        for i in tqdm(range(3150), desc="iterating through the dataset to set the samples per each class"):
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

    def __getitem__(self, index: int):
        # extract the path to the sample (using the map between the index and the sample path !!!)
        if self.samples_per_cls_map is not None:
            stop_points = sorted(list(self.samples_per_cls_map.keys()))

            index = bisect(stop_points, index)
            
            if index == len(stop_points):
                index -= 1

            sp = stop_points[index]

            sample_image:torch.Tensor = self._ds[self.samples_per_cls_map[sp] + index - sp][0]
        else:
            sample_image = self._ds[index]

        if sample_image.shape[0] == 1:
            sample_image = torch.concat([sample_image for _ in range(3)], dim=0)

        augs1, augs2 = self.__set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 

        return s1, s2


    def __len__(self) -> int:
        return self._len
