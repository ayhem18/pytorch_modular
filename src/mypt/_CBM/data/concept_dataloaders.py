"""
This script contains functionalities to prepare the train, validation, and target dataloaders, depending on the given parameters
separating the implementation details from the rest of the training code
"""
import torch
import torchvision.transforms as tr 

from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

from .Clip_label_generation import ClipLabelGenerator
from .datasets import conceptDataset as cd 
from .datasets import uniformConceptDataset as ucd
from .datasets import binaryConceptDataset as bcd
from .datasets import generatedConceptDataset as gcd

from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu

def concept_label_representation1(train_dir: Union[str, Path], 
                         val_dir: Union[str, Path], 
                         target_dir: Optional[Union[str, Path]], 
                         similarity: str, 
                         concepts: Union[str, Dict[str, List[str]]], 
                         image_transformation: tr,
                         remove_existing: bool, 
                         seed: int, 
                         batch_size: int,
                         num_workers: int,
                         shuffle:bool=True
                         ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # depending on the value of the similarity argument, create a Clip generator with using either cosine similarity or dot product
    label_generator = ClipLabelGenerator(similarity_as_cosine=(similarity == 'cosine'))
    
    # create train and validation datasets
    train_ds = cd.ConceptDataset(root=train_dir, 
                            concepts=concepts, 
                            debug=False, 
                            label_generation_batch_size=batch_size,
                            remove_existing=remove_existing,
                            # don't forget to pass the transformation used for Resnet
                            image_transform=image_transformation,
                            label_generator=label_generator)

    # make sure the batch size is at most half of the dataset
    train_batch_size = min(len(train_ds) // 2 - 2, batch_size)
    train_batch_size += 1 if (train_batch_size % len(train_ds) == 1) else 0

    # create a generator to make sure the code is reproducible
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    # create the respective data loaders
    train_dl = DataLoader(train_ds, 
                        batch_size=train_batch_size,
                        shuffle=shuffle,
                        generator=train_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False) # we can't have persistent workers True when setting num_workers to '0'
                        )


    val_ds = cd.ConceptDataset(root=val_dir, 
                            concepts=concepts, 
                            debug=False,
                            label_generator=label_generator,
                            label_generation_batch_size=batch_size,
                            image_transform=image_transformation,
                            remove_existing=remove_existing, 
                            )

    val_batch_size = min(len(val_ds) // 2, batch_size)

    target_ds, target_dl = None, None
    if target_dir is not None:
        # create the dataset
        target_ds = cd.ConceptDataset(root=target_dir, 
                        concepts=concepts, 
                        debug=False,
                        label_generator=label_generator,
                        label_generation_batch_size=batch_size,
                        image_transform=image_transformation,
                        remove_existing=remove_existing, 
                        )


    # increase the batch size until neither the number of items in the validation of target datasets are of modulo 1
    while True: 
        if len(val_ds) % val_batch_size == 1 or (target_ds is not None and len(target_ds) % val_batch_size == 1):
            val_batch_size += 1
        else:
            break

    # set the validation dataloader
    val_gen = torch.Generator()
    val_gen.manual_seed(seed)
    val_dl = DataLoader(val_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=val_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False) 
                        )

    
    # set the target dataloader
    if target_dir is not None:
        target_gen = torch.Generator()
        target_gen.manual_seed(seed)
        target_dl = DataLoader(target_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=target_gen, 
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )
    
    return train_dl, val_dl, target_dl


def concept_label_representation2(train_dir: Union[str, Path], 
                         val_dir: Union[str, Path], 
                         target_dir: Optional[Union[str, Path]], 
                         similarity: str, 
                         concepts: Union[str, Dict[str, List[str]]], 
                         image_transformation: tr,
                         remove_existing: bool, 
                         seed: int,
                         batch_size: int,
                         num_workers: int,
                         shuffle: bool = True
                         ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # depending on the value of the similarity argument, create a Clip generator with using either cosine similarity or dot product
    label_generator = ClipLabelGenerator(similarity_as_cosine=(similarity == 'cosine'))

    # create train and validation datasets
    train_ds = ucd.UniformConceptDataset(root=train_dir, 
                            concepts=concepts, 
                            debug=False, 
                            label_generation_batch_size=batch_size,
                            remove_existing=remove_existing,
                            # don't forget to pass the transformation used for Resnet
                            image_transform=image_transformation,
                            label_generator=label_generator)

    train_batch_size = min(len(train_ds) // 2 - 2, batch_size)
    train_batch_size += 1 if (len(train_ds) % batch_size == 1) else 0

    # create a generator to make sure the code is reproducible
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    # create the respective data loaders
    train_dl = DataLoader(train_ds, 
                        batch_size=train_batch_size,
                        shuffle=shuffle, 
                        generator=train_gen,
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False) # we can't have persistent workers True when setting num_workers to '0'
                        )

    val_ds = ucd.UniformConceptDataset(root=val_dir, 
                            concepts=concepts, 
                            debug=False,
                            label_generator=label_generator,
                            label_generation_batch_size=batch_size,
                            image_transform=image_transformation,
                            remove_existing=remove_existing, 
                            )

    val_batch_size = min(len(val_ds) // 2, batch_size)

    target_ds, target_dl = None, None
    if target_dir is not None:
        # create the dataset
        target_ds = ucd.UniformConceptDataset(root=target_dir, 
                        concepts=concepts, 
                        debug=False,
                        label_generator=label_generator,
                        label_generation_batch_size=batch_size,
                        image_transform=image_transformation,
                        remove_existing=remove_existing, 
                        )

    # increase the batch size until neither the number of items in the validation of target datasets are of modulo 1
    while True: 
        if len(val_ds) % val_batch_size == 1 or (target_ds is not None and len(target_ds) % val_batch_size == 1):
            val_batch_size += 1
        else:
            break

    # set the validation dataloader
    val_gen = torch.Generator()
    val_gen.manual_seed(seed)
    val_dl = DataLoader(val_ds, 
                        batch_size=val_batch_size,
                        shuffle=False, 
                        generator=val_gen,
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )

    
    # set the target dataloader
    if target_dir is not None:
        target_gen = torch.Generator()
        target_gen.manual_seed(seed)
        target_dl = DataLoader(target_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=target_gen, 
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False)                       
                        )
    
    return train_dl, val_dl, target_dl


def concept_label_representation3(train_dir: Union[str, Path], 
                         val_dir: Union[str, Path], 
                         target_dir: Optional[Union[str, Path]], 
                         similarity: str, 
                         concepts: Union[str, Dict[str, List[str]]], 
                         image_transformation: tr,
                         remove_existing: bool, 
                         seed: int,
                         top_k: int,
                         batch_size: int,
                         num_workers: int,
                         shuffle:bool=True
                         ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    pu.seed_everything(seed)
    train_ds = bcd.BinaryConceptDataset(root=train_dir, 
                                        concepts=concepts,
                                        similarity=similarity,
                                        top_k=top_k,
                                        image_transform=image_transformation,
                                        label_generation_batch_size=batch_size,
                                        remove_existing=remove_existing,
                                        )

    train_batch_size = min(len(train_ds) // 2 - 2, batch_size)
    # avoid having a batch size of only 1 sample 
    train_batch_size += 1 if (len(train_ds) % batch_size == 1) else 0

    # create a generator to make sure the code is reproducible
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    # create the respective data loaders
    train_dl = DataLoader(train_ds, 
                        batch_size=train_batch_size,
                        shuffle=shuffle,
                        generator=train_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False) # we can't have 'persistent_workers' set to True if 'num_workers' is '0'
                        )

    val_ds = bcd.BinaryConceptDataset(root=val_dir, 
                                        concepts=concepts,
                                        similarity=similarity,
                                        top_k=top_k,
                                        image_transform=image_transformation,
                                        label_generation_batch_size=batch_size,
                                        remove_existing=remove_existing,
                                        )
    
    target_ds, target_dl = None, None
    if target_dir is not None:
        # create the dataset
        target_ds = bcd.BinaryConceptDataset(root=target_dir,
                                            concepts=concepts,
                                            similarity=similarity,
                                            top_k=top_k,
                                            image_transform=image_transformation,
                                            label_generation_batch_size=batch_size,
                                            remove_existing=remove_existing,
                                            )

    val_batch_size = min(len(val_ds) // 2, batch_size)

    # increase the batch size until neither the number of items in the validation of target datasets are of modulo 1
    while True: 
        if len(val_ds) % val_batch_size == 1 or (target_ds is not None and len(target_ds) % val_batch_size == 1):
            val_batch_size += 1
        else:
            break

    # set the validation dataloader
    val_gen = torch.Generator()
    val_gen.manual_seed(seed)
    val_dl = DataLoader(val_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=val_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )

    
    # set the target dataloader
    if target_dir is not None:
        target_gen = torch.Generator()
        target_gen.manual_seed(seed)
        target_dl = DataLoader(target_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=target_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )
    
    return train_dl, val_dl, target_dl


def concept_label_representation4(train_dir: Union[str, Path], 
                         val_dir: Union[str, Path], 
                         target_dir: Optional[Union[str, Path]], 
                         image_transformation: tr,
                         remove_existing: bool, 
                         seed: int, 
                         batch_size: int,
                         num_workers: int,
                         ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    train_ds = gcd.GeneratedConceptDataset(root=train_dir, 
                                           image_transform=image_transformation,
                                           label_generation_batch_size=batch_size, 
                                           remove_existing=remove_existing)    

    # make sure the batch size is at most half of the dataset
    train_batch_size = min(len(train_ds) // 2 - 2, batch_size)
    train_batch_size += 1 if (train_batch_size % len(train_ds) == 1) else 0

    # create a generator to make sure the code is reproducible
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    # create the respective data loaders
    train_dl = DataLoader(train_ds, 
                        batch_size=train_batch_size,
                        shuffle=True,
                        generator=train_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False) # we can't have persistent workers True when setting num_workers to '0'
                        )

    val_ds = gcd.GeneratedConceptDataset(root=val_dir, 
                                           image_transform=image_transformation,
                                           label_generation_batch_size=batch_size, 
                                           remove_existing=remove_existing)    

    val_batch_size = min(len(val_ds) // 2, batch_size)

    target_ds, target_dl = None, None
    if target_dir is not None:
        # create the dataset
        target_ds = gcd.GeneratedConceptDataset(root=target_dir,
                                           image_transform=image_transformation,
                                           label_generation_batch_size=batch_size,
                                           remove_existing=remove_existing)    

    # increase the batch size until neither the number of items in the validation of target datasets are of modulo 1
    while True: 
        if len(val_ds) % val_batch_size == 1 or (target_ds is not None and len(target_ds) % val_batch_size == 1):
            val_batch_size += 1
        else:
            break

    # set the validation dataloader
    val_gen = torch.Generator()
    val_gen.manual_seed(seed)
    val_dl = DataLoader(val_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=val_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False) 
                        )

    
    # set the target dataloader
    if target_dir is not None:
        target_gen = torch.Generator()
        target_gen.manual_seed(seed)
        target_dl = DataLoader(target_ds, 
                        batch_size=val_batch_size,
                        shuffle=False,
                        generator=target_gen, 
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )
    
    return train_dl, val_dl, target_dl


def set_dataloaders(train_dir: Union[str, Path], 
                    val_dir: Union[str, Path], 
                    target_dir: Optional[Union[str, Path]], 
                    representation: int,
                    similarity: str, 
                    concepts: Union[str, Dict[str, List[str]]], 
                    image_transformation: tr,
                    seed: int,
                    batch_size: int, 
                    num_workers: int,
                    remove_existing: bool = False,
                    shuffle: bool = True,
                    **kwargs)  -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    # make sure to     
    train_dir = dirf.process_path(train_dir)
    val_dir = dirf.process_path(val_dir)
    target_dir = dirf.process_path(target_dir)

    if similarity not in ['cosine', 'dot']:
        raise ValueError(f"The 'similarity' argument is expected to be either {['cosine', 'dot']}. Found: {similarity}")

    if representation not in [1, 2, 3, 4]:
        raise ValueError(f"The 'representation' argument is expected to belong to one of these values: {[1, 2, 3]}. Found: {representation}") 

    if representation == 1: 
        return concept_label_representation1(train_dir=train_dir, 
                    val_dir=val_dir, 
                    target_dir=target_dir,
                    similarity=similarity, 
                    concepts=concepts,
                    image_transformation=image_transformation, 
                    remove_existing=remove_existing, 
                    seed=seed, 
                    batch_size=batch_size, 
                    num_workers=num_workers,
                    shuffle=shuffle)
    
    if representation == 2:
        return concept_label_representation2(train_dir=train_dir, 
                    val_dir=val_dir, 
                    target_dir=target_dir,
                    similarity=similarity, 
                    concepts=concepts,
                    image_transformation=image_transformation, 
                    remove_existing=remove_existing, 
                    seed=seed, 
                    batch_size=batch_size, 
                    num_workers=num_workers,
                    shuffle=shuffle)

    if representation == 4:
        return concept_label_representation4(train_dir=train_dir,
                                             val_dir=val_dir, 
                                             target_dir=target_dir,
                                            image_transformation=image_transformation, 
                                            remove_existing=remove_existing, 
                                            seed=seed, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers
                                            )


    top_k = kwargs['kwargs']['top_k']
    
    return concept_label_representation3(train_dir=train_dir, 
                                        val_dir=val_dir, 
                                        target_dir=target_dir, 
                                        similarity=similarity,
                                        concepts=concepts,
                                        image_transformation=image_transformation, 
                                        remove_existing=remove_existing, 
                                        seed=seed, 
                                        batch_size=batch_size, 
                                        top_k=top_k,
                                        num_workers=num_workers, 
                                        shuffle=shuffle
                                        )

