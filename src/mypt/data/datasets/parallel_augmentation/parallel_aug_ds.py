"""
This script contains the implementation of a general dataset object designed for Constrastive Learning Parallel Augmentation approaches 
(https://arxiv.org/pdf/2002.05709, https://arxiv.org/abs/2103.03230) for example
"""

import os
from typing import Union, List, Tuple, Optional
from pathlib import Path

from ....code_utilities import directories_and_files as dirf
from .parallel_aug_abstract import AbstractParallelAugsDs


class ParallelAugDs(AbstractParallelAugsDs):
    def __init__(self, 
                root: Union[str, Path],
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_data_augs: List,
                image_extensions:Optional[List[str]]=None,
                seed: int=0):
        
        super.__init__(
                output_shape=output_shape,
                augs_per_sample=augs_per_sample,
                sampled_data_augs=sampled_data_augs,
                uniform_data_augs=uniform_data_augs,
                image_extensions=image_extensions,
                seed=seed)

        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS

        self.root = dirf.process_path(root,
                                    file_ok=False,
                                    dir_ok=True,
                                    # the directory should contain only images files (no dirs)
                                    condition=lambda x: all([os.path.isfile(os.path.join(x, p)) and os.path.splitext(os.path.join(x, p))[-1] in self.im_exts for p in os.listdir(x)]),
                                    error_message=f'The root directory is expected to contains only image files and no directories')

        # create a mapping between a numerical index and the associated sample path for O(1) access time (on average...)
        self.idx2path = None
        # set the mapping from the index to the sample's path
        self._prepare_idx2path()
        
        # count the number of samples once
        self.data_count = len(os.listdir(root))


    def _prepare_idx2path(self):
        # make sure to sort files names (for cross-platform reproducibilty )
        samples = sorted(os.listdir(self.root))
        self.idx2path = dict([(index, os.path.join(self.root, fn)) for index, fn in enumerate(samples)])


    def __getitem__(self, index: int):
        # extract the path to the sample (using the map between the index and the sample path !!!)
        sample_image = self.load_sample(self.idx2path[index])   
        augs1, augs2 = self.__set_augmentations()
        s1, s2 = augs1(sample_image), augs2(sample_image) 
        return s1, s2


    def __len__(self) -> int:
        if self.data_count == 0:
            raise ValueError(f"Please make sure to update the self.data_count field in the constructor to have the length of the dataset precomputed !!!!. Found: {self.data_count}")
        return self.data_count
