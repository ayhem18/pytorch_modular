"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, itertools, torch, json
import torchvision.transforms as tr

from typing import Union, List, Dict 
from pathlib import Path
from tqdm import tqdm

from .abstractConceptDataset import AbstractConceptDataset
from ..Clip_label_generation import ClipLabelGenerator
from ....code_utilities.pytorch_utilities import cleanup

class ConceptDataset(AbstractConceptDataset):
    def _prepare_labels(self,
                        batch_size: int, 
                        debug: bool = False) -> None:
        """
        This function iterates through the 'self.root' directory creating the concepts labels for each sample
        and saving them in separate directories within 'self.root'. Such process is conducted to avoid
        repeated inference during training.

        Returns:
        """
        # start by freeing up any available GPU memory
        cleanup()
        
        class_names = [f for f in sorted(os.listdir(self.root)) if f in self.classes]

        for folder_name in tqdm(class_names, desc='concept labels for each class: representation 1'):
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

                # generate the concept labels
                batch_labels = self.label_generator.generate_image_label(batch_file_paths, self.concepts_features, debug_memory=debug)

                for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels_path):
                    # save the concept label
                    torch.save(label_tensor, label_file_path)

    def __init__(self,
                 root: Union[str, Path],
                 concepts: Union[Dict[str, List[str]], List[str]],
                 label_generation_batch_size: int = 512,
                 image_transform: tr = None,
                 label_generator=None,
                 remove_existing: bool = True, 
                 debug: bool = False):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        super().__init__(root,
                          image_transform=image_transform, 
                          remove_existing=remove_existing)
        
        if isinstance(concepts, (Path, str)):
            with open(concepts, 'r') as f:
                concepts = json.load(f)
            
        # process the concepts
        concepts = concepts if isinstance(concepts, list) else list(itertools.chain(*(concepts.values())))
        # filter duplicate concepts
        self.concepts = list(set(concepts))

        # save the number of initial folders in 'root'
        initial_num_folders = len(os.listdir(self.root))
        
        # create the label generator
        self.label_generator = label_generator if label_generator is not None else ClipLabelGenerator()
        # save the features of the concepts as they will be used for the label generation of every image
        self.concepts_features = self.label_generator.encode_concepts(self.concepts)
        

        _labels_ready = False
        while not _labels_ready:
            try:
                self._prepare_labels(batch_size=label_generation_batch_size)
                _labels_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                label_generation_batch_size /= 2 

        # at this point the number of directories should have doubled
        assert not remove_existing or len(os.listdir(root)) == 2 * initial_num_folders, \
            "The number of directories is not doubled !!!"
