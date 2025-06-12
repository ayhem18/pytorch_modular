"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, itertools, torch, json
import torchvision.transforms as tr

from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict 


from mypt.shortcuts import P
from ....code_utils.pytorch_utils import cleanup
from .abstract_concept_dataset import AbstractConceptDataset
from .concepts_generation.Clip_label_generation import ClipLabelGenerator



class BasicConceptDataset(AbstractConceptDataset):
    
    def _prepare_labels(self, batch_size: int) -> None:
        """_summary_

        Args:
            batch_size (int): _description_
        """
        cleanup()

        for i in range(0, len(self._ds), batch_size):
            batch_file_paths = [self._ds.idx2path[idx] for idx in range(i, min(i + batch_size, len(self._ds)))]

            batch_concept_labels = super()._verify_full_batch_label_generation(batch_file_paths)

            if batch_concept_labels is None:
                # this means that these samples have been encountered before: no need to generate concept labels
                continue 

            batch_labels = self.label_generator.generate_image_label(batch_file_paths, self.concepts_features)
            
            for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels):
                torch.save(label_tensor, label_file_path)



    def __init__(self,
                 root: P,
                 concepts: Union[Dict[str, List[str]], List[str]],
                 filenames: Iterable[str],
                 label_dir: P,
                 label_generation_batch_size: int = 512,
                 transforms: Optional[tr.Compose] = None,
                 label_generator = None,
                 remove_existing: bool = True,
                 image_extensions: Optional[List[str]] = None,
                 label_suffix: str = '_concept_label',
                 ):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        super().__init__(root=root,
                         filenames=filenames,
                         label_dir=label_dir,
                         transforms=transforms,
                         remove_existing=remove_existing,
                         image_extensions=image_extensions,
                         label_suffix=label_suffix)
        
        if isinstance(concepts, (Path, str)):
            with open(concepts, 'r') as f:
                concepts = json.load(f)
            
        # process the concepts
        concepts = concepts if isinstance(concepts, list) else list(itertools.chain(*(concepts.values())))
        # filter duplicate concepts
        self.concepts = list(set(concepts))

        
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
                label_generation_batch_size = int(label_generation_batch_size / 2)


