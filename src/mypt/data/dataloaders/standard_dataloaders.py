"""
This script contain functionalities related to data loading shared among different tasks
"""
import torch, warnings
import numpy as np

from functools import partial
from warnings import warn
from typing import List, Optional

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from ...code_utilities.pytorch_utilities import set_worker_seed


def initialize_train_dataloader(dataset_object: Dataset, 
                        seed: int,
                        batch_size: int,
                        num_workers: int,
                        weights: Optional[List[float] | np.ndarray] = None,
                        drop_last: bool = True,
                        warning: bool = True,
                        pin_memory:bool = False
                        ) -> DataLoader:
    """This function initializes a dataloader making sure the data loading is reproducible across runs. 

    Args:
        dataset_object (Dataset): The dataset to load / train on
        seed (int): The seed to assume reproducibility
        batch_size (int): 
        num_workers (int): the number of sub-processes 
        drop_last (bool, optional): whether to drop the last batch. Defaults to True.

    Returns:
        DataLoader: A dataloader assumed to load training data for a model
    """

    if not drop_last and warning:
        warn(f"The parameter 'drop_last' is set to False. Depending on the size of the dataset and the batch size." 
             f"this might lead to reported metrisc lower than the actual ones.")


    # under the rare case that the size of the dataset is smaller than the batch size, this function will return an empty 
    # dataloader when the 'drop_last' argument is set to True

    # let's add a check to that: 
    if len(dataset_object) <= batch_size and drop_last:
        raise ValueError(f"Found a dataset with size {len(dataset_object)} and a batch size : {batch_size}. When setting the 'drop_last' param to True, the dataloader will be empty. Make sure there is no issue with dataset...")

    # if the size of the dataset is still less than the batch size, raise a warning (as it might signal a problem in the dataset size calculation)
    if len(dataset_object) <= batch_size:
        warnings.warn(message=f"Found a dataset with size {len(dataset_object)} and a batch size : {batch_size}. Make sure there is no issue with dataset...")

    # create the generator of the dataloader
    dl_train_gen = torch.Generator()
    dl_train_gen.manual_seed(seed)

    if weights is not None:
        
        # make sure the length of the weights is equal to the number of samples
        # apparently such a constraint is not explicitly enforced by Pytorch but might lead to several issues later down the line
        if len(weights) != len(dataset_object):
            raise ValueError((f"The length of the `weights` iterable does not match the size of the dataset:" 
                             "Found: {len(weights)} weights and {len(dataset_object)} samples "))
        
        # in case of concrete weights, initialize a sampler and pass a generator to the sampler
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset_object), replacement=True, generator=dl_train_gen)
        # set the generator to None
        generator = None

        # Pytorch implementation requires the "shuffle" and "sampler" arguments to be exclusive, setting the samples => setting the "shuffle" to None
        shuffle = None

    else:
        # set the sampler to None
        sampler = None
        # and then set the generator
        generator = dl_train_gen
        # Pytorch implementation requires the "shuffle" and "sampler" arguments to be exclusive
        shuffle = True 

    if num_workers != 0:
        dl_train = DataLoader(dataset=dataset_object, 
                            shuffle=shuffle,  
                            # the train data loader must be shuffled !!
                            drop_last=drop_last, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            worker_init_fn=partial(set_worker_seed, seed=seed), # this function ensures reproducibility between runs in multi-process setting 
                            generator=generator, 
                            persistent_workers=True,
                            sampler=sampler)
        return dl_train

    # if the number of workers is set to '0', then the parameter 'pin_memory' will be set to True to improve performance
    # make sure to warn the user
    if warning:
        warn(message=f"the 'num_workers' argument is set to 0. The dataloader will be run by the main process !!!")

    dl_train = DataLoader(dataset=dataset_object, 
                        shuffle=shuffle, # the train dataloader must be shuffled not to hurt performance
                        drop_last=drop_last, 
                        batch_size=batch_size, 
                        num_workers=0, 
                        pin_memory=pin_memory,
                        generator=generator,
                        sampler=sampler
                        )
    
    return dl_train



def initialize_val_dataloader(dataset_object: Dataset, 
                        seed: int,
                        batch_size: int,
                        num_workers: int,
                        warning:bool=True,
                        collate_fn=None,
                        ) -> DataLoader:

    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    if num_workers != 0:
        dl = DataLoader(dataset=dataset_object, 
                            shuffle=False, # generally there is no point in shuffing the validation data
                            batch_size=batch_size,
                            drop_last=False, 
                            num_workers=num_workers, 
                            worker_init_fn=partial(set_worker_seed, seed=seed), # this function is used to ensure reproducibility between runs in multi-process setting 
                            generator=dl_gen,
                            collate_fn=collate_fn, 
                            persistent_workers=True)
        return dl

    # if the number of workers is set to '0', then the parameter 'pin_memory' will be set to True to improve performance
    # make sure to warn the user
    if warning:
        warn(message=f"the 'num_workers' argument is 0.")

    dl = DataLoader(dataset=dataset_object, 
                        shuffle=False,
                        drop_last=False, 
                        batch_size=batch_size, 
                        num_workers=0,   
                        pin_memory=False,
                        collate_fn=collate_fn) 
    
    return dl


