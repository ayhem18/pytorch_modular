"""
This script contain functionalities related to data loading shared among different tasks
"""
import torch

from functools import partial
from warnings import warn

from torch.utils.data import Dataset, DataLoader
from ...code_utilities.pytorch_utilities import set_worker_seed


def initialize_train_dataloader(dataset_object: Dataset, 
                        seed: int,
                        batch_size: int,
                        num_workers: int,
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

    dl_train_gen = torch.Generator()
    dl_train_gen.manual_seed(seed)

    if num_workers != 0:
        dl_train = DataLoader(dataset=dataset_object, 
                            shuffle=True, # the train dataloader must be shuffled not to hurt performance
                            drop_last=drop_last, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            worker_init_fn=partial(set_worker_seed, seed=seed), # this function ensures reproducibility between runs in multi-process setting 
                            generator=dl_train_gen, 
                            persistent_workers=True)
        return dl_train

    # if the number of workers is set to '0', then the parameter 'pin_memory' will be set to True to improve performance
    # make sure to warn the user
    if warning:
        warn(message=f"the 'num_workers' argument is set to 0. The dataloader will be run by the main process !!!")

    dl_train = DataLoader(dataset=dataset_object, 
                        shuffle=True, # the train dataloader must be shuffled not to hurt performance
                        drop_last=drop_last, 
                        batch_size=batch_size, 
                        num_workers=0, 
                        generator=dl_train_gen,
                        pin_memory=pin_memory)
    
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
                            collate_fn=collate_fn)
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


