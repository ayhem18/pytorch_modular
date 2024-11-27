"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import os, torch, itertools, json

import torchvision.transforms as tr

from math import ceil
from typing import Union, List, Dict
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from .abstractConceptDataset import AbstractConceptDataset
from ..Clip_label_generation import ClipLabelGenerator
from ....code_utilities import pytorch_utilities as pu

class BinaryConceptDataset(AbstractConceptDataset):
    def __init__(self,
                 root: Union[str, Path],
                 concepts: Union[Dict[str, List[str]]],
                 similarity: str,
                 percent_close_classes: Union[float, int],
                 top_k: int = 1,
                 label_generation_batch_size: int = 512,
                 image_transform: tr = None,
                 remove_existing: bool = True):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        # make sure the 'percent_close_classes' is a float in the range ]0, 1]
        
        if percent_close_classes <= 0 or percent_close_classes > 1:
            raise ValueError(f"The 'percent_close_classes' argument must be in the range ]0, 1]. Found: {percent_close_classes}")

        super().__init__(root, 
                         image_transform=image_transform, 
                         remove_existing=remove_existing)
        
        # Adding the capability of ignoring / discaring concepts depending on the class they are associated with
        # requires having the class <-> concepts information

        if not isinstance(concepts, (str, Path, Dict)):
            raise NotImplementedError(f"The dataset requires having the concepts class information which can be passed only through a 'file' or a 'dict'")

        # handle the case where the 'concepts' are passed as a path to a json file
        if isinstance(concepts, (str, Path)):
            # Make sure the path is to a .json file
            if os.path.splitext(os.path.basename(concepts))[-1] != ".json":
                raise ValueError(f"a path to a file was passed to the Dataset. The file must be a json file. Found: {os.path.basename(concepts)}")
            
            with open(concepts, 'r') as f:
                concepts = json.load(f)        

        # the 'concepts' object 
        if not isinstance(concepts, Dict):
            # the 'concepts' object at this point must be a Dictionary
            raise ValueError(f"The 'concepts' object must be a dictionary !!!. Found: {type(concepts)}")


        flattened_concepts = list(itertools.chain(*[concepts[c] for c in self.classes]))
        self.concepts = flattened_concepts 

        # create a small map between the class index and the indices of the concepts associated with that class
        self.cls_2_concepts_indices = {}
        
        total = 0
        for c in self.classes:
            self.cls_2_concepts_indices[self.class_to_idx[c]] = (total, total + len(concepts[c]))
            total += len(concepts[c])

        if similarity not in ['cosine', 'dot']: 
            raise ValueError(f"Please make sure the argument 'similarity' is in {['cosine', 'dot']}. Found: {similarity}")
        self.sim = similarity

        # create the label generator
        self.clip_generator = ClipLabelGenerator(similarity_as_cosine=self.sim == 'cosine')
        self.concepts_features = self.clip_generator.encode_concepts(concepts=self.concepts)

        # find the 'close' classes for each class
        self.percent_close_classes = percent_close_classes
        self.num_close_classes = int(ceil(len(self.classes) * self.percent_close_classes))

        self._find_close_classes()

        # save the number of initial folders in 'root'
        initial_num_folders = len(os.listdir(self.root))

        self.top_k = top_k 

        # let's compute the thresholds for each concept
        t_ready = False
        batch_size = label_generation_batch_size
        while not t_ready:
            try:
                self._compute_thresholds(concepts_encoded=self.concepts_features, 
                                        batch_size=batch_size)
                t_ready = True
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # empty the cache
                pu.cleanup()
                batch_size = int(batch_size / 1.2)

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


    def _find_close_classes(self):
        """
        This method finds the closes classes for each class using the CLIP text encoder.
        it initializes a field 'close_classes_per_cls' where each class index is mapped to the indices for the 'close' classes.
        
        Additionally, it computes the mapping between 'L_i' and the classes for which 'L_i' is considered a close class.
        """

        # encode each of the classes using the CLIP text encoder
        classes_encoded = self.clip_generator.encode_concepts(concepts=self.classes)
        
        # compute the similarities between the classes
        classes_sims = classes_encoded @ classes_encoded.T
        
        if classes_sims.shape != (len(self.classes), len(self.classes)):
            raise ValueError(f"Make sure the classes similarities are computed correctly !!. Expected shape: {(len(self.classes), len(self.classes))}. Found: {classes_sims.shape}")

        # compute the number of classes that will be considered close for each class

        # choose the top classes for each class
        _, indices = torch.topk(input=classes_sims, k=self.num_close_classes, dim=1, largest=True)

        if indices.shape != (len(self.classes), self.num_close_classes):
            raise ValueError(f"Make sure the indices of the close classes are computed correctly")

        self.close_classes_per_cls = {i : indices[i, :].tolist() for i in range(len(self.classes))}
        self.reverse_close_classes = defaultdict(lambda : [])

        for cls_index, close_classes in self.close_classes_per_cls.items():
            for j in close_classes:
                self.reverse_close_classes[j].append(cls_index)
        
        self.reverse_close_classes


    def _find_cls_associated_concepts(self, cls: Union[str, int]) -> List[int]:
        
        cls_index = cls if isinstance(cls, int) else self.class_to_idx[cls]
        # filter the concepts
        associated_classes = self.reverse_close_classes[cls_index]
        concepts_indices = []
        for ac in associated_classes:
            concepts_start, concepts_end = self.cls_2_concepts_indices[ac]
            concepts_indices.extend(list(range(concepts_start, concepts_end)))

        return concepts_indices


    def _class_concept_similarities(self, 
                                    cls_name: str,
                                    concepts_encoded: torch.Tensor, 
                                    batch_size: int) -> torch.Tensor :
        """
        Given the embeddings of all concepts, this method computes the similarity matrix between the samples with the 'cls_name' label 
        and all concepts with an extra twist 


        only the concepts that satisfy the following condition will have their actual similarity computed, the result will be associated with -torch.inf 

        {concepts 'c' such that the class associated with 'c': L cls_name belongs to S(L)}

        The shape is a similarity matrix of shape (samples, num_concepts_satisfy_condition)

        Args:
            cls_name (str): The class name
            concept_feature (torch.Tensor): an encoding of a concept
            batch_size (int): the size of the batch to be used if the samples cannot be load to the memory at once

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: the similarities between all the samples and the encoding of the given concept: batched or unbatched
        """
        
        class_path = os.path.join(self.root, cls_name)
        class_images = sorted(os.listdir(class_path))
        class_images = [os.path.join(class_path, i) for i in class_images]

        num_concepts, _ = concepts_encoded.shape

        concepts_indices = self._find_cls_associated_concepts(cls=cls_name)

        if len(concepts_indices) == 0:
            raise ValueError("some class is associated with no classes. Each class must be at least associated with itself !!!")

        try:
            # try to pass the data at once
            class_concept_sims = self.clip_generator.generate_image_label(images=class_images, 
                                                                          concepts_features=concepts_encoded, 
                                                                          apply_softmax=False)
            if class_concept_sims.shape != (len(class_images), num_concepts): 
                raise ValueError((f"The similarities between the class samples and the concept should be of shape:" 
                                 f"{(len(class_images),  num_concepts)}. Found: {class_concept_sims.shape}"))

            # return class_concept_sims

        except (MemoryError, torch.cuda.OutOfMemoryError):
            # empty the GPU cache
            pu.cleanup()

            # iterate with a batch size
            class_concept_sims = torch.cat([
                                            self.clip_generator.generate_image_label(images=class_images[i : i + batch_size], 
                                                                                     concepts_features=concepts_encoded, 
                                                                                     apply_softmax=False) 
                                            for i in range(0, len(class_images), batch_size)
                                            ], dim=0) 
            if class_concept_sims.shape != (len(class_images), num_concepts): 
                raise ValueError((f"The similarities between the class samples and the concept should be of shape:" 
                                 f"{(len(class_images),  num_concepts)}. Found: {class_concept_sims.shape}"))


        # before returing the similarity matrix
        # the similarities between the class and the non-selected concepts should be set to - inf
        
        # we need a mask to set the similarities to zero for the unrelated concepts
        mask = torch.zeros(*class_concept_sims.shape)
        mask[:, concepts_indices] = 1

        # the -inf should be added only to unrelated concepts
        # so the minus_inf_tensor should be set to 0 for the related concepts
        minus_inf_tensor = torch.full(class_concept_sims.shape, fill_value=-float('inf'))
        minus_inf_tensor[:, concepts_indices] = 0

        similarity_matrix = class_concept_sims * mask + minus_inf_tensor

        return similarity_matrix


    def _compute_thresholds(self, 
                            concepts_encoded: torch.Tensor,                             
                            batch_size: int) -> torch.Tensor:
        """ 
        This method computes the similarity thresholds for each concept. For a given concept associated with a label 'L_i', the concept similarity threshold
        should consider only samples belongs to labels 'L' such that L belong to the 'close classes' of the label 'L_i'

        """        
        
        # iterate through the classes
        top_k_similarities = torch.stack([torch.topk(self._class_concept_similarities(cls_name=cls_name, 
                                                                                    concepts_encoded=concepts_encoded, 
                                                                                    batch_size=batch_size), 
                                                    k=self.top_k, 
                                                    dim=0, 
                                                    largest=True)[0]
                                            for cls_name in tqdm(self.classes, desc='class concept similarities for each class')], 
                                        dim=0)
        
        # make sure the shape is as expected
        if top_k_similarities.shape != (len(self.classes), self.top_k, concepts_encoded.shape[0]):
            raise ValueError(f"The matrix of similarities between a concept and all classes is of the wrong shape: Expected:"
                             f"{len(self.classes), self.top_k, concepts_encoded.shape[0]}. Found: {top_k_similarities.shape}")

        # make sure to convert the -inf to + inf
        is_inf_mask = torch.isinf(top_k_similarities)
        top_k_similarities[is_inf_mask] = float('inf')

        # torch.min returns a couple with the values and the indices when called using the 'dim' keyword argument
        thresholds=  torch.min(top_k_similarities[:, self.top_k - 1, :], dim=0)[0]

        if torch.any(torch.isinf(thresholds)):
            raise ValueError(f"there are infinity values in the thresholds tensor. Make sure the thresholding process is correct")
        # make sure to unsqueeze if needed
        thresholds = thresholds.unsqueeze(dim=0) if thresholds.ndim == 1 else thresholds

        if thresholds.shape != (1, concepts_encoded.shape[0]):
            raise ValueError(f"The thresholding has not been performed successfully !! Expected shape: {(1, concepts_encoded.shape[0])} else ")

        self.concepts_thresholds = thresholds
    
    def _generate_concept_labels(self, cls: Union[int, str], images: List[Union[str, Path]]) -> torch.Tensor:

        concepts_indices = self._find_cls_associated_concepts(cls=cls)

        image_concept_similarities = self.clip_generator.generate_image_label(images=images, 
                                                                              concepts_features=self.concepts_features, 
                                                                              apply_softmax=False)
        
        if image_concept_similarities.shape != (len(images), len(self.concepts)):
            raise ValueError(f"Make sure the similarities between the images and the concepts are calculated correctly." 
                             f"Expected: {(len(images), len(self.concepts))} .Found: {image_concept_similarities.shape}") 

        thresholds = torch.cat([self.concepts_thresholds for _ in range(len(images))], dim=0)
        assert thresholds.shape == image_concept_similarities.shape, "The thresholds and the similarities should be of the exact same shape."
        concept_labels = ((image_concept_similarities - thresholds) >= 0).to(torch.float32)         
        
        # make sure to set the values for the unrelated concepts to 0
        mask = torch.zeros(*concept_labels.shape)
        mask[:, concepts_indices] = 1
        final_concept_labels = concept_labels * mask    

        return final_concept_labels

    def _prepare_labels(self, batch_size: int) -> None:
        """
        It is important to keep in mind that
        This function iterates through the 'self.root' directory

        """
        # start by freeing up any previously occupied GPU memory
        pu.cleanup()

        # extract the class names

        class_names = [f for f in sorted(os.listdir(self.root)) if f in self.classes]

        for folder_name in tqdm(class_names, desc='concept labels for each class: representation 3'):
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

                batch_labels = self._generate_concept_labels(cls=folder_name, images=batch_file_paths)

                # make sure the labels are indeed binary
                if not torch.all(torch.logical_or(input=(batch_labels == 1), other=(batch_labels == 0))):
                    raise ValueError(f"the concept label should contain the values 1 or 0.")

                if batch_labels.shape != (len(batch_file_paths), len(self.concepts)):
                    raise ValueError(f"The labels are expected to be of the shape: {(len(batch_file_paths), len(self.concepts))}. Found: {batch_labels.shape}")

                for label_tensor, label_file_path in zip(batch_labels, batch_concept_labels_path):
                    # save the concept label
                    torch.save(label_tensor, label_file_path)

        