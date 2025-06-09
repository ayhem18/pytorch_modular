"""
This script contains the implementation of a concrete Concept Datasets that maps each class into a randomly generated binary vector. 
The vectors are designed to be highly discriminative and nearly linearly-seperable
"""

import os, torch, itertools

import torchvision.transforms as tr

from typing import Union, List, Dict
from pathlib import Path
from tqdm import tqdm

from .abstractConceptDataset import AbstractConceptDataset
from ....code_utilities import pytorch_utilities as pu


class GeneratedConceptDataset(AbstractConceptDataset):
    def __init__(self, 
        root: Union[str, Path],
        image_transform: tr,
        block_per_cls: int = 25, 
        cls_threshold: float = 0.95,
        out_cls_min_threshold: float=0.005, # 50%
        out_cls_max_threhsold: float=0.2, # 0.5 %
        label_generation_batch_size:int=512,
        remove_existing: bool = True,
        seed: int = 69):    
        super().__init__(root,  
                        image_transform=image_transform,
                        remove_existing=remove_existing)

        # save the number of initial folders in 'root'
        initial_num_folders = len(os.listdir(self.root))

        # save the thresholds
        self.block_per_cls = block_per_cls
        # as we know the size of the block for each class, the concept label will be of size 'block_per_cls' * num_classes
        self.cl_dim = self.block_per_cls * len(self.classes) 

        self.cls_threshold = cls_threshold
        self.out_cls_min_t = out_cls_min_threshold
        self.out_cls_max_t = out_cls_max_threhsold

        # prepare the data; experim
        _labels_ready = False
        batch_size = label_generation_batch_size
        while not _labels_ready:
            try:
                self._prepare_labels(batch_size=batch_size)
                _labels_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # empty the cache
                pu.cleanup()
                batch_size = int(batch_size / 1.2)
            
        # at this point the number of directories should have doubled
        assert not remove_existing or len(os.listdir(root)) == 2 * initial_num_folders, \
            "The number of directories is not doubled !!!"

    def _generate_concept_labels(self, n:int , cls_index: int):
        # first generate a random value of numbers between 0 and 1
        initial_random_vec = torch.rand(n, self.cl_dim)
        
        cls_center_index = self.block_per_cls * cls_index + self.block_per_cls // 2
        max_distance_to_center = max(self.cl_dim - cls_center_index, cls_center_index)

        def out_cls_threshold_by_index(index: int):
            # since the class main indics have already been processed, we will assign them zero
            if self.block_per_cls * cls_index <= index < self.block_per_cls * (cls_index + 1):
                return self.cls_threshold

            # the thresold should be higher the closer the index is to the center of the class
            # an index at the border of the block will have a value very close to self.out_cls_max_t
            return (1 - abs(index - cls_center_index) / max_distance_to_center) * self.out_cls_max_t + self.out_cls_min_t

        # the next step is to generate the thesholds for the non class blocks
        mask = torch.tensor([[out_cls_threshold_by_index(i) for i in range(self.cl_dim)] for _ in range(n)])

        final_random_vector = ((mask - initial_random_vec) >= 0).to(torch.float32)

        return final_random_vector

    def _prepare_labels(self, batch_size: int) -> None:
        """
        It is important to keep in mind that
        This function iterates through the 'self.root' directory

        """
        # start by freeing up any occupied GPU memory
        pu.cleanup()

        # extract

        class_names = [f for f in sorted(os.listdir(self.root)) if f in self.classes]

        for folder_name in tqdm(class_names, desc='concept labels for each class: representation 4'):
            # at this point we know the current folder is a class folder
            cls_path = os.path.join(self.root, folder_name)
            # extract the paths to all the images
            image_paths = [os.path.join(self.root, folder_name, file) for file in sorted(os.listdir(cls_path))]
            
            # create the label folder for the current class if needed
            cls_label_folder_path = os.path.join(self.root, f'{folder_name}_{self.concept_label_ending}')
            os.makedirs(cls_label_folder_path, exist_ok=True)

            # create the concept labels in batches
            for i in range(0, len(image_paths), batch_size):
                # extract the paths in the current batch
                batch_file_paths = image_paths[i:i + batch_size]
                # extract the paths to the corresponding concept labels
                batch_concept_labels_path = [self._sample_to_concept_label(sample_path) for sample_path in batch_file_paths]

                # at this point, either all of them should be present or none of them
                concept_labels_present = [os.path.exists(bclp) for bclp in batch_concept_labels_path]

                # we should make sure that either all files in the batch are associated with a concept label or none of them:
                # this ensures reproducibility
                if any(concept_labels_present) and not all(concept_labels_present):
                    raise ValueError(f"Some files in the batch have concept labels and some do not. Please make sure the code is reproducible!!!")
                
                if all(concept_labels_present):
                    # this means that these samples have been encountered before: no need to generate concept labels
                    continue 

                batch_labels = self._generate_concept_labels(len(batch_file_paths),cls_index=self.class_to_idx[folder_name])

                # make sure the labels are indeed binary
                if not torch.all(torch.logical_or(input=(batch_labels == 1), other=(batch_labels == 0))):
                    raise ValueError(f"the concept label should contain the values 1 or 0.")

                for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels_path):
                    # save the concept label
                    torch.save(label_tensor, label_file_path)

        