"""
This script contains the implementation of a general dataset object designed for Constrastive Learning Parallel Augmentation approaches 
(https://arxiv.org/pdf/2002.05709, https://arxiv.org/abs/2103.03230) for example
"""

import os
from typing import Union, List, Tuple, Optional
from pathlib import Path

from ....code_utilities import directories_and_files as dirf
from .parallel_aug_abstract import AbstractParallelAugsDs


class ParallelAugDirDs(AbstractParallelAugsDs):
    def __init__(self, 
                root_dir: Union[str, Path],
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_augs_before: List,
                uniform_augs_after: List,
                image_extensions:Optional[List[str]]=None,
                seed: int=0):
        
        super().__init__(
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_augs_before= uniform_augs_before,
                uniform_augs_after=uniform_augs_after,
                seed=seed)

        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS

        # the root directory can have any structure as long as 
        # it contains only image data
        self.root_dir = dirf.process_path(root_dir, 
                                      file_ok=False,
                                      dir_ok=True,
                                      condition=lambda x: dirf.dir_contains_only_types(x, valid_extensions=image_extensions), # contains only image data
                                      error_message=f'The root directory is expectd to contain only image data: specifically those extensions: {image_extensions}'
                                      )
        
        # create a mapping between a numerical index and the associated sample path for O(1) access time (on average...)
        self.idx2path = None
        # set the mapping from the index to the sample's path
        self._prepare_idx2path()


    def _prepare_idx2path(self):
        # define a dictionary
        idx2path = {}
        counter = 0

        for r, _, files in os.walk(self.root_dir):
            for f in files:
                file_path = os.path.join(r, f)
                idx2path[counter] = file_path
                counter += 1
        # sorted the samples for reproducibility
        paths = sorted(list(idx2path.values()))
        self.idx2path = dict([(i, p) for i, p in enumerate(paths)])
        self.data_count = len(self.idx2path)

        # I initially thought of shuffling the indices since the directory might represent an image classification task
        # and the consecutive indices belong to the same class. However, as I can see this is also the case
        # for Pytorch built-in datasets
        # and the shuffling part should be done at the dataloader level


    def __getitem__(self, index: int):
        # extract the path to the sample (using the map between the index and the sample path !!!)
        sample_image = self.load_sample(self.idx2path[index])   
        augs1, augs2 = self._set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 
        return s1, s2


    def __len__(self) -> int:
        if self.data_count == 0:
            raise ValueError(f"Please make sure to update the self.data_count field in the constructor to have the length of the dataset precomputed !!!!. Found: {self.data_count}")
        return self.data_count
