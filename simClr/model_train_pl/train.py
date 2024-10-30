"""
This script contais the training code for the SimClrWrapper PL model
"""

# imports
import torch, os
import numpy as np
import pytorch_lightning as L

# from imports
from clearml import Task, Logger
from typing import Union, Optional, Tuple, List, Callable, Dict, Iterable
from pathlib import Path
from collections import defaultdict
from itertools import chain
from sklearn.cluster import KMeans

# torch and pl imports
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import CombinedLoader

# imports from the mypt packge
from mypt.data.dataloaders.standard_dataloaders import initialize_val_dataloader
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.data.datasets.parallel_augmentation.parallel_aug_dir import ParallelAugDirDs
from mypt.shortcuts import P
from mypt.similarities.cosineSim import CosineSim

# imports from the directory
from .simClrWrapper import ResnetSimClrWrapper
from .set_ds import _set_data, _set_imagenette_ds_debug

from .constants import (_TEMPERATURE, _OUTPUT_DIM, _OUTPUT_SHAPE, 
                        ARCHITECTURE_IMAGENETTE, ARCHITECTURE_FOOD_101, 
                        _NUM_FC_LAYERS, _DROPOUT,
                        TRACK_PROJECT_NAME) 

def _set_logging(use_logging: bool, 
                 run_name: str, 
                 return_task: bool, 
                 init_task:bool=True,
                 **kwargs) -> Union[Tuple[Task, Logger], Optional[Logger]]:

    if use_logging:
        # create a clearml task object
        if init_task:
            task = Task.init(project_name=TRACK_PROJECT_NAME,
                            task_name=run_name,
                            **kwargs
                            )
            logger = task.get_logger()
        else:
            task = Task.create(project_name=TRACK_PROJECT_NAME,
                            task_name=run_name,
                            )
            logger = task.get_logger()
    else:
        logger = None
        task = None
    
    if return_task:
        return task, logger
    
    return logger


def _set_ckpnt_callback(val_dl: Optional[DataLoader], 
                        num_epochs:int, 
                        val_per_epoch:int,
                        log_dir: P) -> ModelCheckpoint:

    if val_dl is not None:
        # if the validation dataset is passed, make sure
        # the val_per_epoch argument is less than half the number of epochs
        # so that at least 2 checkpoints are considered     
        if num_epochs // 2 < val_per_epoch:
            raise ValueError("Passing a validation dataset while setting the 'val_per_epoch' less than the number of epochs will eventually raise an error !!!") 


    ckpnt_split = 'val' if val_dl is not None else 'train'
    # if the validation dataset is passed, then we need to have al least one validation epoch before checkpointing
    # otherwise update the checkpoint every epoch (we need the best training loss...)
    every_n_epochs = val_per_epoch if val_dl is not None else 1
    monitor = f"{ckpnt_split}_epoch_loss"

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=1, 
                                        monitor=monitor, # the checkpointing process might depend on either the train on the validation data
                                        mode='min', 
                                        every_n_epochs=every_n_epochs, 
                                        # save the checkpoint with the epoch and validation loss
                                        filename= ckpnt_split + '-{epoch:02d}-'+ '{' + monitor + ':06f}',
                                        # auto_insert_metric_name=True,
                                        save_on_train_epoch_end=(val_dl is None) # if there is a validation set then check after the val epoch
                                        )

    return checkpnt_callback

def _find_best_ckpnt(log_dir: P) -> P:
    # this function assumes the checkpoints were 
    # created using the checkpoint callback
    # defined above
    min_val_loss = float('inf')

    for ckpnt_org in os.listdir(log_dir):
        ckpnt, _ = os.path.splitext(ckpnt_org)
        _, _, val_loss = ckpnt.split('-')
        _, val_loss = val_loss.split('=')
        val_loss = float(val_loss)
        if val_loss < min_val_loss:
            best_checkpoint = ckpnt_org
            min_val_loss = val_loss

    return best_checkpoint


def train_simClr_single_round(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        train_batch_size: int,
        val_batch_size: int,
        samples_weights: Optional[Iterable[float]], # list, np.ndarray, torch.Tensor...

        log_dir: Union[str, Path],
        run_name: str,
        continue_last_task: bool,

        debug_augmentations: List,

        initial_checkpoint: Optional[P] = None,
        output_shape:Tuple[int, int]=None, 
        
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,

        use_logging: bool = True,
        
        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ) -> ResnetSimClrWrapper:    

    if output_shape is None:
        output_shape = _OUTPUT_SHAPE
    
    # process the loggin directory
    log_dir = dirf.process_path(log_dir, file_ok=False, dir_ok=True)

    # set the logger
    logger = _set_logging(use_logging=use_logging, 
                        run_name=run_name, 
                        return_task=False,
                        # according to  https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
                        # continue_last_task will take the maximum iteration value for each plot and use an incremented counter
                        continue_last_task=continue_last_task)

    simClr_train_dl, simClr_val_dl, debug_train_dl = _set_data(
                                                    # dataset arguments
                                                    train_data_folder=train_data_folder,
                                                    val_data_folder=val_data_folder, 
                                                    dataset="imagenette",
                                                    output_shape=output_shape,
                                                    # dataloader arguments
                                                    train_batch_size=train_batch_size,
                                                    val_batch_size=val_batch_size,
                                                    samples_weights = samples_weights,

                                                    num_train_samples_per_cls=num_train_samples_per_cls,
                                                    num_val_samples_per_cls=num_val_samples_per_cls,
                                                    seed=seed)

    checkpnt_callback = _set_ckpnt_callback(val_dl=simClr_val_dl, 
                                            num_epochs=num_epochs, 
                                            val_per_epoch=val_per_epoch, 
                                            log_dir=log_dir)

    lr = 0.3 * (train_batch_size / 256) # according to the paper's formula

    # define the wrapper
    wrapper = ResnetSimClrWrapper(
                                  # model parameters
                                  input_shape=(3,) + output_shape, 
                                  output_dim=_OUTPUT_DIM,
                                  num_fc_layers=_NUM_FC_LAYERS,

                                  #logging parameters
                                  logger=logger,
                                  log_per_batch=3,
                                  val_per_epoch=val_per_epoch,
                                  # loss parameters 
                                  temperature=_TEMPERATURE,
                                  debug_loss=True,

                                  # learning 
                                  lrs=lr,
                                  num_epochs=num_epochs,
                                  num_warmup_epochs=num_warmup_epochs,

                                  debug_embds_vs_trans=True, 
                                  debug_augmentations=debug_augmentations,

                                  # more model parameters
                                  dropout=_DROPOUT,
                                  architecture=ARCHITECTURE_IMAGENETTE,
                                  freeze=False
                                  )

    # get the default device
    device = pu.get_default_device()

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu' if 'cuda' in device else 'cpu',     ## I don't have acess to TPUs at the time of writing this code)))
                        devices=1, # not sure of the aftermath of setting this parameter to 'auto' (1 is the safest choice for now)
                        # strategy='ddp',
                        logger=False, # since I am not using any of the supported logging options, logger=False + using the self.log function would do the job...
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=1 if len(simClr_train_dl) < 10 else 5,
                        callbacks=[checkpnt_callback]
                        )


    trainer.fit(model=wrapper,
                train_dataloaders=simClr_train_dl,
                val_dataloaders=CombinedLoader([simClr_val_dl, debug_train_dl], 
                                                    mode='sequential'), 
                ckpt_path=initial_checkpoint
                )

    # find the checkpoint
    best_ckpnt = _find_best_ckpnt(log_dir)

    return wrapper, best_ckpnt


def evaluate_augmentations(augmentations: List, 
                           get_augmentation_name: Callable,
                           augmentation_scores: Dict,
                           num_categories: int
                           ) -> Dict:
    
    for a in augmentations:
        if get_augmentation_name(a) not in augmentation_scores:
            raise ValueError(f"Make sure the callable to generate the augmentation names is the same used in the generation of the augmentation scores")

    # compute the mean, std, and the quantiles, 0.25, 0.5, 0,75
    augs_stats = [(get_augmentation_name(a), 
                        [
                         np.mean(augmentation_scores[get_augmentation_name(a)]).item(), 
                         np.std(augmentation_scores[get_augmentation_name(a)]).item()
                        ] 
                        + 
                        np.percentile(augmentation_scores[get_augmentation_name(a)], [25, 50, 75]).tolist()
                        )
                    for a in augmentations
                ]

    augs_samples = np.asarray([v for _, v in augs_stats])

    # use the statistics to cluster 
    kmeans = KMeans(n_clusters=num_categories, 
                # metric='euclidean', 
                random_state=0,)

    labels: np.ndarray = kmeans.fit_predict(augs_samples)

    # having the labels prepared, it is time to order the augmentation categories by difficulty
    labels_unique = np.unique(labels)
    
    labels_unique = sorted(labels_unique, # the unique label 
                            key=lambda l: np.mean(augs_samples[labels == l.item(), -2]).item(), # for each label 'l' index the augs_samples with the labels and average the median... 
                            reverse=True) # higher average implies easier augmentation

    # at this point the labels are sorted by difficulty
    augmentations_per_difficulty = defaultdict(lambda :[])

    label2difficulty = dict((l.item(), index) for index, l in enumerate(labels_unique))

    for augmentation_index, aug_label in enumerate(labels):
        augmentations_per_difficulty[label2difficulty[aug_label.item()]].append(augmentations[augmentation_index])
    
    return augmentations_per_difficulty


def calculate_weights(augmentations_per_difficulty: Dict, 
                    dataloader: DataLoader,
                    model: torch.nn.Module,
                    process_model_output: Callable,
                    ):
    
    # set the device
    device = pu.get_default_device()
    # move the model to the device
    model = model.to(device)

    # extract the difficulty levels
    difficulties = sorted(list(augmentations_per_difficulty.keys()))
    
    # extract the augmentations ordered in difficulty
    augs = list(
        chain.from_iterable([augmentations_per_difficulty[i] for i in difficulties])
    )

    # the k-th hardest category is assocaited 1 / sqrt(k + 1)
    # first calculate the weight similarities mean between a given samples and all the augmentations
    # harder samples should have lower values
    # apply 1 / weights to obtain the final weights

    weights = list(
                    chain.from_iterable([
                                [np.exp(- (d - len(difficulties) // 2) ** 2) for _ in range(len(augmentations_per_difficulty[d]))] 
                                for d in difficulties
                                ])
                )
    

    dataset_result = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_embds = process_model_output(model.forward(batch))
            batch_res = []

            for t in augs:
                batch_t = t(batch)
                batch_t_embds = process_model_output(model.forward(batch_t)) 
                # compute the similarities between samples and their augmented versions
                sample_aug_similarities = torch.diagonal(CosineSim().forward(batch_embds, batch_t_embds)).unsqueeze(-1)
                batch_res.append(sample_aug_similarities)

            batch_res = torch.concat(batch_res, dim=1) # concatenate horizontally
            dataset_result.append(batch_res)

        # concatenate all the batches vertically to build the dataset
        dataset_result = torch.concat(dataset_result, dim=0)

    # convert to numpy
    dataset_result = dataset_result.cpu().numpy()

    # make sure to clip anything negative to zero
    dataset_result = np.clip(dataset_result, 
                             a_min=0.0001, 
                             a_max=1)

    samples_weights = dataset_result @ np.expand_dims(np.asarray(weights), axis=-1)
    
    # samples with larger values are more likely to be easier
    # hence should be assigned smaller probabilities
    # inverse the values and pass them as weights for the datalaoder
    final_weights = (1 / samples_weights).squeeze()
    
    return final_weights


def train_simClr(
        train_data_folder:Union[str, Path],          
        val_data_folder:Optional[Union[str, Path]],
        dataset: str,

        num_epochs: int, 
        train_batch_size: int,
        val_batch_size: int,

        log_dir: Union[str, Path],
        run_name: str,

        debug_augmentations: List,
        total_count: int = 3,
        
        output_shape:Tuple[int, int]=None, 
        num_warmup_epochs:int=10,
        val_per_epoch: int = 3,
        seed:int = 69,

        use_logging: bool = True,

        num_train_samples_per_cls: Union[int, float]=None, # the number of training samples per cls
        num_val_samples_per_cls: Union[int, float]=None, # the number of validation samples per cls
        ):
    
    # initialize a counter to keep track of the number of runs
    counter = 0
    last_ckpnt = None
    samples_weights = None # set to None on the first round

    while counter < total_count:
        round_log_dir = os.path.join(log_dir, f'round_{counter + 1}')
        # call the train_simClrWrapper function
        wrapper, last_ckpnt = train_simClr_single_round(train_data_folder=train_data_folder,
                                            val_data_folder=val_data_folder,
                                            dataset=dataset, 
                                            num_epochs=num_epochs,

                                            train_batch_size=train_batch_size,
                                            val_batch_size=val_batch_size,
                                            samples_weights=samples_weights,

                                            log_dir=round_log_dir,
                                            run_name=run_name,
                                            continue_last_task=(counter > 0), # only reuse the task_id on later iterations

                                            debug_augmentations=debug_augmentations,
                                            initial_checkpoint=last_ckpnt, # set the last checkpoint as a starting point for the new round (initially set to None)
                                            
                                            output_shape=output_shape, 
                                            num_warmup_epochs=num_warmup_epochs * int(counter == 0), ## use warm-up epochs only on the first round
                                            val_per_epoch=val_per_epoch,
                                            seed=seed,

                                            use_logging=use_logging,

                                            num_train_samples_per_cls=num_train_samples_per_cls,
                                            num_val_samples_per_cls=num_val_samples_per_cls
                                            )
        
        augs_per_difficulty = evaluate_augmentations(
                                            augmentations=debug_augmentations,
                                            get_augmentation_name=pu.get_augmentation_name, 
                                            augmentation_scores=wrapper.augmentation_scores,
                                            num_categories=3)
        
        dataloader = _set_imagenette_ds_debug(train_data_folder=train_data_folder, 
                                    output_shape=(200, 200), 
                                    batch_size=1024, 
                                    num_train_samples_per_cls=None)
    
        samples_weights = calculate_weights(augmentations_per_difficulty=augs_per_difficulty, 
                                        dataloader=dataloader, 
                                        model=wrapper.model, 
                                        process_model_output=lambda x: x[1])


        # make sure to increase the counter !!!!
        counter += 1
