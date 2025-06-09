"""
This script contains the implementation of a simple linear prob, that receives the concept representations and predict the classes.
"""
import os
import torch
import numpy as np
import pytorch_lightning as L
import torchvision.transforms as tr
import wandb
import pickle

from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Iterator, Tuple, Optional, Dict, Any, List, Union
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from functools import partial
from tqdm import tqdm 

from ....linearBlocks.classification_head import ExponentialClassifier
from ....code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from ...data.datasets.conceptDataset import ConceptDataset

WANDB_PROJECT_NAME = 'CBM-UDA'

class LinearProb(nn.Module):
    def __init__(self, 
                 representation_dim: int, 
                 num_classes:int, 
                 num_layers: int,
                 dropout: float,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_prob = ExponentialClassifier(num_classes=num_classes, 
                                                 in_features=representation_dim,
                                                 num_layers=num_layers,
                                                 dropout=dropout
                                                 )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_prob.forward(x)
    
    def __str__(self):
        return self.linear_prob.__str__()
    
    def __repr__(self):
        return self.linear_prob.__repr__() 
    
    def children(self) -> Iterator[nn.Module]:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.feature_extractor
        return self.linear_prob.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.linear_prob.modules()
    
    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.linear_prob.named_children()

class LinearProbWrapper(L.LightningModule):
    def __init__(self, 
                concept_representation_dim: int, 
                num_classes: int,
                num_layers: int,   
                optimizer_class: callable,
                dropout: Optional[float] = None,
                learning_rate: Optional[float] = 10 ** -4,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,   
                ):
        super().__init__()
        self.concept_representation_dim = concept_representation_dim
        self.num_classes = num_classes        

        self.lr = (learning_rate if learning_rate is not None else 10 ** -4)
        self.opt = optimizer_class
        self.opt_args = optimizer_keyargs

        # save the scheduler arguments
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_keyargs

        self._model = LinearProb(representation_dim=self.concept_representation_dim, 
                                num_layers=num_layers,
                                num_classes=num_classes,
                                dropout=dropout)        

        # create some fields to save the model's metrics: total loss, classification loss, concept loss, accuracy
        self.train_epoch_losses = []
        self.train_epoch_accs = []

        # the same for validation losses
        self.val_epoch_losses = []
        self.val_epoch_accs = []

        # add some fields to track the performance of the model on the target domain: this fields are used mainly
        # to account for the uneven split in batches when calculating the target accuracy      
        self.batch_target_accuracies = []
        self.batch_target_losses = []

        # these fields are used mainly to save the losses after each validation epoch (if the target dataloader is passed of course)
        self.target_epoch_losses = []
        self.target_epoch_accs = []

        self.save_hyperparameters()

    def forward(self, 
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._model.forward(x.to(torch.float).to(pu.get_module_device(self)))

    def _forward_pass(self, batch, reduce_loss: bool = True):
        # first seperate the input
        concept_representations, class_labels = batch
        logits = self._model.forward(concept_representations)
        # calculate accuracy
        class_predictions = torch.argmax(logits, dim=1)
        accuracy = torch.mean((class_predictions.to(torch.long) == class_labels.to(torch.long)).to(torch.float)).cpu().item()
        # simple classification loss
        loss_obj = nn.CrossEntropyLoss(reduction=('mean' if reduce_loss else 'none')).forward(input=logits, target=class_labels)
        return loss_obj, class_predictions, accuracy

    def on_train_epoch_start(self) -> None:
        # this is carried out to avoid sharing values across epochs
        self.train_loss = 0
        self.train_acc = 0
        self.train_batch_count = 0
        
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        loss, _, accuracy = self._forward_pass(batch)

        self.log_dict({"train_loss": loss.cpu().item(), 
                       "train_accuracy": round(accuracy, 5)})
        
        # make sure to add each of the metrics to the variable that is supposed to store it
        self.train_loss += loss.cpu().item()
        self.train_acc += accuracy  
        self.train_batch_count += 1
        return loss 
    
    def on_train_epoch_end(self) -> None:
        # at the end of each epoch, average the results and save them
        # the total loss
        self.train_epoch_losses.append(self.train_loss / self.train_batch_count)
        # the train accuracy
        self.train_epoch_accs.append(self.train_acc / self.train_batch_count)

    def _val_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Since the main validation step function will operate on potentially 2 dataloaders: 
        1. the validation split dataloader
        2. the target domain dataloader
        This function is called to calculate the model's metric on the validation split
        Args:
            batch: Tuple[torch.Tensor, torch.Tensor]: the validation step batch: should contain a batch of images and class labels
        """
        # sanity check: 
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 3 and 
                isinstance(batch[0], torch.Tensor) and 
                isinstance(batch[1], torch.Tensor) and
                isinstance(batch[2], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the validation split in the source domain.")

        loss, _, accuracy = self._forward_pass(batch)

        self.log_dict({"val_loss": loss.cpu().mean().item(), 
                       "val_accuracy": round(accuracy, 5)},
                        add_dataloader_idx=False)
        
        # make sure to add each of the metrics to the variable that is supposed to store it
        self.val_loss += loss.cpu().item()
        self.val_acc += accuracy  
        self.val_batch_count += 1
        return loss 

    def _target_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Since the main "validation_step" function will operate on potentially 2 dataloaders: 
        1. the validation split dataloader
        2. the target domain dataloader
        This function is called to calculate the model's accuracy on the target domain
        Args:
            batch: Tuple[torch.Tensor, torch.Tensor]: the validation step batch: should contain a batch of images and class labels
        """
        # sanity check
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 3 
                and isinstance(batch[0], torch.Tensor) 
                and isinstance(batch[1], torch.Tensor) 
                and isinstance(batch[2], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the target domain.")

        _, class_labels = batch
        target_loss, _, target_acc = self._forward_pass(batch, reduce_loss=False)
        # the target accuracy is calculated as a mean over the batch (since batches are not necessarily of the same number of elements, we calculate the sum)
        # and divide by the total number of elements
        self.batch_target_accuracies.extend([target_acc for _ in range(len(class_labels))])
        self.batch_target_losses.append(target_loss.cpu())

    def on_validation_epoch_start(self) -> None:
        self.val_loss = 0
        self.val_acc = 0
        self.val_batch_count = 0

    def validation_step(self, 
                        batch: Union[List, Tuple], 
                        batch_idx: int, 
                        *args: Any):        
        """
        This function handels both data from the validation split on the source domain or data from the target domain. 
        Depending on the format the method will call subsequent methods to handel each type of data accordingly. 
        
        The function expects the data to be passed in a CombinedLoader with 'sequential' mode assuring all data is passed 
        and without cycling

        Args:
            batch (Union[List[torch.Tensor, torch.Tensor, torch.Tensor], List[torch.Tensor, torch.Tensor]]): the data
            batch_idx (int): the index of the validation batch
        """
        if len(args) == 0:
            return self._val_step(batch=batch, batch_idx=batch_idx)

        if len(args) != 1:
            raise ValueError(f"the function is expecting at most 1 arguments in 'args'. Found: {args}")

        iterable_id = args[0]
        
        # then call the corresponding function depending on the format
        if iterable_id == 0:
            return self._val_step(batch=batch, batch_idx=batch_idx)
        
        if iterable_id == 1:
            # call the self._target_step with the target domain data
            return self._target_step(batch=batch)

        raise ValueError(f"The iteratble id argument can be either {0} or {1}. Found: {iterable_id}")

    def on_validation_epoch_end(self) -> None:
        """
        The first step can is to average the different losses and accuracies on the validation split.

        The second step is to calculate the accuracy of the model on the target daomin (if it is passed)
        Since we would like to make sure the target accuracy is calculated correctly. We need to account for the uneven split of batches.
        This function will iterate through self.target_Accuracies and self.target_losses fields and calculate the correct average accuracy and average loss
        by summing both metrics and dividing by the total number of elements in the target split
        """ 
        # first step: always applicable

        # at the end of each epoch, average the results and save them
        # the total loss
        self.val_epoch_losses.append(self.val_loss / self.val_batch_count)
        # the validation accuracy
        self.val_epoch_accs.append(self.val_acc / self.val_batch_count)

        # 2nd step: check if a target dataloader is passed
        if (len(self.batch_target_accuracies) + len(self.batch_target_losses) == 0):
            return
        
        # first convert the target losses field into a single tensor
        target_losses = torch.cat(self.batch_target_losses, dim=0)

        if len(target_losses) != len(self.batch_target_accuracies):
            raise ValueError(("Please make sure the number of elements in the 'self.target_losses' and the 'self.target_accuracies' are the same.\n"
                             f"\nFound: {len(target_losses)} losses and {len(self.target_accuracies)} accuracies"))
        

        loss, accuracy = target_losses.mean().cpu().item(), np.mean(self.batch_target_accuracies)

        # log the results
        self.log_dict({"target_loss": loss, "target_accuracy": accuracy}, 
                      add_dataloader_idx=False)

        # clear the fields to use in the next epoch
        self.batch_target_losses.clear()
        self.batch_target_accuracies.clear()

        # make sure to save these values (and not just log them ...)
        self.target_epoch_accs.append(accuracy)
        self.target_epoch_losses.append(loss)

    def configure_optimizers(self):
        # create the optimizer the call depends on whether there are scheduler keyarguments passed
        if self.opt_args is None:
            optimizer = self.opt(self._model.parameters(), lr=self.lr)
        else:
            optimizer = self.opt(self._model.parameters(), lr=self.lr, **self.opt_args)
        
        # create lr scheduler
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer=optimizer, **self.scheduler_args)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
    

from ...data.concept_dataloaders import set_dataloaders

def _set_dataloaders(train_dir: Union[str, Path], 
                     val_dir: Union[str, Path], 
                     target_dir: Union[str, Path], 
                     representation: int,
                     similarity: str,
                     seed: int, 
                     image_transformation: tr,
                     batch_size: int,
                     num_workers: int,
                    ) -> Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
                               DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
                               Optional[DataLoader[Tuple[torch.Tensor, torch.Tensor]]]]:
    train_dl, val_dl, target_dl = set_dataloaders(train_dir=train_dir,
                                                  val_dir=val_dir, 
                                                  target_dir=target_dir,
                                                  representation=representation, 
                                                  similarity=similarity, 
                                                  seed=seed,
                                                  image_transformation=image_transformation,
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  remove_existing=False)
    # the idea here is to modify the collate funciton to drop the image and leave only the concept representations and class labels
    train_dl.collate_fn = lambda _, x, y : (x, y)
    val_dl.collate_fn = lambda _, x, y : (x, y)

    if target_dl is not None:
        target_dl.collate_fn = lambda _, x, y : (x, y)

    return train_dl, val_dl, target_dl

def train_classifier_wrapper(
                    wrapper: LinearProbWrapper,
                    train_dir:Union[Path, str],
                    val_dir: Union[Path, str],
                    log_dir: Union[str, Path],
                    image_transformation: tr,
                    target_dir: Union[Path, str] = None,
                    run_name: str = None,
                    num_workers: int = 0,
                    batch_size: int = 1024,
                    num_epochs: int = 10, 
                    wandb_logger: WandbLogger = None, 
                    repeat_config: bool = False, 
                    seed: int = 69, 
                    ) -> LinearProbWrapper:

    pu.seed_everything(seed=seed)
    if wandb_logger is None:
        # wandb details for logging
        wandb.init(project=WANDB_PROJECT_NAME, 
                name=run_name)
        
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                                log_model=False, 
                                save_dir=log_dir, 
                                name=run_name)

    train_dir = dirf.process_path(train_dir, file_ok=False, 
                                       condition=dirf.image_dataset_directory, 
                                       error_message=dirf.image_dataset_directory_error(train_dir))
    
    val_dir = dirf.process_path(val_dir, file_ok=False,
                                     condition=dirf.image_dataset_directory, 
                                     error_message=dirf.image_dataset_directory_error(val_dir))
  
    target_dir = dirf.process_path(target_dir, file_ok=False,
                                        condition=dirf.image_dataset_directory, 
                                        error_message=dirf.image_dataset_directory_error(target_dir))                                        

    log_dir = dirf.process_path(log_dir, file_ok=False)

    # make sure the log_dir passed does not exist already
    if os.path.exists(log_dir) and len(os.listdir(log_dir)) != 0:
        if repeat_config: 
            counter = 1
            new_log_dir = os.path.join(Path(log_dir).parent, f'{os.path.basename(log_dir)}_{counter}')
            
            while os.path.exists(new_log_dir):
                counter += 1
                new_log_dir =  os.path.join(Path(log_dir).parent, f'{os.path.basename(log_dir)}_{counter}')
            
            log_dir = new_log_dir
        else:
            raise ValueError(f"Please make sure the logging directory is empty or not yet created. To avoid overwritting pervious checkpoints !!")


    train_dl, val_dl, target_dl = _set_dataloaders(train_dir=train_dir, 
                                                   val_dir=val_dir, 
                                                   target_dir=target_dir,
                                                   image_transformation=image_transformation,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers, 
                                                   seed=seed)

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_cls_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='classifier-{epoch:02d}-{val_cls_loss:06f}')

    # define the trainer
    log_step = 1 if len(train_dl) < 10 else 10
    trainer = L.Trainer(
                        accelerator='gpu',
                        devices=1,
                        logger=wandb_logger,
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=3,
                        log_every_n_steps=log_step,
                        callbacks=[checkpnt_callback])

    # make sure to pass the correct form depending on the target directory 
    # val_dataloaders = ({_VALIDATION_DATALOADER_KEY: val_dl, _TARGET_DATALOADER_KEY: target_dl}) if target_dl is not None else (val_dl)
    val_dataloaders = (CombinedLoader([val_dl, target_dl], 
                                      mode='sequential')) if target_dl is not None else (val_dl)

    trainer.fit(model=wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dataloaders,
                )

    # find the checkpoint with the lowest validation loss
    # iterate through the log directory
    best_checkpoint, min_val_loss = None, float('inf')
    
    for ckpnt_org in os.listdir(log_dir):
        ckpnt, _ = os.path.splitext(ckpnt_org)
        _, _, val_loss = ckpnt.split('-')
        _, val_loss = val_loss.split('=')
        val_loss = float(val_loss)
        if val_loss < min_val_loss:
            best_checkpoint = ckpnt_org
            min_val_loss = val_loss
    
    # make sure to explicity end the run here
    wandb.finish()

    print(f'Best checkpoint: {best_checkpoint}')

    # at this point we can access all the metrics
    metrics = {
            "train_losses": wrapper.train_epoch_losses, 
            "train_accs": wrapper.train_epoch_accs, 

            # for validation and target losses, the metrics of the sanity check are also included 
            # so we will discard them
            "val_losses": wrapper.val_epoch_losses[1:], 
            "val_accs": wrapper.val_epoch_accs[1:], 

            # make sure the code works both for the cases with and without target data [1:] will raise an error if the list is empty
            "target_losses": (wrapper.target_epoch_losses[1:] if target_dir is not None else []),
            "target_accs": (wrapper.target_epoch_accs[1:] if target_dir is not None else []),
            }

    # make sure to save the metrics after determining the best checkpoint (so the code above won't break)
    with open(os.path.join(log_dir, 'metrics.obj'), 'wb') as f: 
        pickle.dump(metrics, f)

    return os.path.join(log_dir, best_checkpoint)


def linear_prob_predict(wrapper: LinearProbWrapper, 
            data_dir: Union[str, Path],
            concepts: Union[List[str], Dict[str, List[str]]], 
            image_transformation: tr,
            seed: int,
            batch_size: int = 1024,
            num_workers: int = 10, 
            ) -> float:
    
    # set the model to the 'eval' mode
    wrapper.eval()
    
    # the data directly might contain concept labels which are not image data. Hence we cannot use the default ImageFolder dataset
    # provided by Pytorch. Since we are using the concepts labels in the inference, we can use any Concept dataset
    dataset = ConceptDataset(root=data_dir,
                             concepts=concepts,
                             image_transform=image_transformation, 
                             remove_existing=False, 
                             label_generation_batch_size=batch_size,    
                             )

    # since the cbm uses batch normalization it throws an error for 1-sample batches, choose a batch size guaranteeing this won't happen
    # increment the batch size until the length of the dataset is not 1 modulo the batch size
    while True: 
        if len(dataset) % batch_size == 1:
            batch_size += 1
        else:
            break
    
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=num_workers, 
                            worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                            collate_fn=lambda _, x, y: (x, y) 
                            )
    total_count = 0
    correct_count = 0

    with torch.no_grad():
        for x, cls_labels in tqdm(dataloader, desc=f'CBM: inference on the dataset'):
            # move the labels to the same device as the model's
            cls_labels = cls_labels.to(pu.get_module_device(wrapper))
            total_count += len(x)
            cls_logits = wrapper.forward(x)
            cls_predictions = torch.argmax(cls_logits, dim=1)
            correct_count += torch.sum((cls_predictions.to(torch.long) == cls_labels.to(torch.long)).to(torch.float)).cpu().item()
    
    accuracy = round(correct_count / total_count, 4)
    return accuracy
