"""
This script contains the implementation of a general dataset object designed for Constrastive Learning Parallel Augmentation approaches 
(https://arxiv.org/pdf/2002.05709, https://arxiv.org/abs/2103.03230) for example
"""
import os, random

import torchvision.transforms as tr
from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple
from pathlib import Path
from PIL import Image

from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu


# for this class to work properly, it is important to know the expected input of each image augmentation
# some of them accept only PIL images, some accept PIL and ndarrys
# some of them accept only tensors..
 
# 0: PIL image
# 1: ndarray
# 2: torch Tensor 
_data_aug_input_type = {tr.ToTensor: [0], 
                        tr.Resize: [0, 2]}



class ParallelAugDs(Dataset):
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
                root: Union[str, Path],
                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_data_augs: List,
                image_extensions:List[str]=None,
                seed: int=0):

        # reproducibiliy provides a much better idea about the performance
        pu.seed_everything(seed=seed)
        if image_extensions is None:
            image_extensions = dirf.IMAGE_EXTENSIONS
        
        self.im_exts = image_extensions

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

        # the output of the resulting data (After augmentation)
        self.output_shape = output_shape

        # make sure each transformation starts by resizing the image to the correct size
        self.sampled_data_augs = sampled_data_augs
        self.uniform_data_augs = uniform_data_augs
        self.augs_per_sample = min(augs_per_sample, len(self.sampled_data_augs))



    def _prepare_idx2path(self):
        # make sure to sort files names (for cross-platform reproducibilty )
        samples = sorted(os.listdir(self.root))
        self.idx2path = dict([(index, os.path.join(self.root, fn)) for index, fn in enumerate(samples)])


    def __getitem__(self, index: int):
        # extract the path to the sample (using the map between the index and the sample path !!!)
        sample_image = self.load_sample(self.idx2path[index])   

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
        

        s1, s2 = augs1(sample_image), augs2(sample_image) 
        # these variables are created for debugging purposes
        n1, n2 = s1.numpy(), s2.numpy()

        return s1, s2


    def __len__(self) -> int:
        if self.data_count == 0:
            raise ValueError(f"Please make sure to update the self.data_count field in the constructor to have the length of the dataset precomputed !!!!. Found: {self.data_count}")
        return self.data_count
