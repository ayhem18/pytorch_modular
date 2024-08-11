import random, torch

import torchvision.transforms as tr

from pathlib import Path 
from typing import Union, List, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST, STL10



class STL10Wrapper(Dataset):
    def __init__(self, root_dir: Union[str, Path], 
                train:bool,

                output_shape: Tuple[int, int],
                augs_per_sample: int,
                sampled_data_augs:List,
                uniform_data_augs: List,
                
                length: int = None) -> None:
        super().__init__()
        
        self._ds = STL10(root=root_dir, 
                         split='train' if train else 'test', 
                         transform=tr.ToTensor(), 
                         download=True)
        
        self._len = length if length is not None else len(self._ds)

        self.output_shape = output_shape

        # make sure each transformation starts by resizing the image to the correct size
        self.sampled_data_augs = sampled_data_augs
        self.uniform_data_augs = uniform_data_augs
        self.augs_per_sample = min(augs_per_sample, len(self.sampled_data_augs))

    def __getitem__(self, index: int):
        # extract the path to the sample (using the map between the index and the sample path !!!)
        sample_image = self._ds[index][0]

        if sample_image.shape[0] == 1:
            sample_image = torch.concat([sample_image for _ in range(3)], dim=0)

        augs1, augs2 = random.sample(self.sampled_data_augs, self.augs_per_sample), random.sample(self.sampled_data_augs, self.augs_per_sample)

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
        return self._len
