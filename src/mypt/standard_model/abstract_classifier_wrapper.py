"""
This script contains the main code to train a standard Image classification model with Resnet as a backbone
"""

import os, torch, pickle, wandb
import numpy as np

import pytorch_lightning as L
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as tr

from pathlib import Path
from typing import Any, Union, Tuple, Optional, Iterable, Dict, List
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from ..code_utilities import directories_and_files as dirf
from ..code_utilities import pytorch_utilities as pu
from ..swag.posteriors.swag import SWAG

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WANDB_PROJECT_NAME = 'CBM-UDA'

class ClassifierWrapper(L.LightningModule):
    def __init__(self, 
                input_shape: Tuple,
                num_classes: int,   
                optimizer_class: callable,
                learning_rate: Optional[Union[float, Iterable[float]]] = 10 ** -4,
                num_vis_images: int = 3,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,
                swag: bool = False
                 ):
        super().__init__()

        self.input_shape = input_shape 
        self.num_classes = num_classes        

        self.lr = (learning_rate if learning_rate is not None else 10 ** -4)
        self.opt = optimizer_class
        self.opt_args = optimizer_keyargs

        # save the scheduler arguments
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_keyargs

        # the concrete class will have to implement those fields
        self.is_swag = swag 
        self._model, self._transform, self._swag_model = None, None, None

        # save the number of imgaes to be logged at each validation step
        self.num_vis_images = num_vis_images

        # create a field that will save all the data to be logged
        self.log_data = pd.DataFrame(data=[], columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

        self.save_hyperparameters()

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

        # create special fields to 
        self.batch_target_accuracies = []
        self.batch_target_losses = []

    def forward(self, 
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function extracts the user_id, item_id, and features in the input
        passes each of them to the corresponding layer
        and then the output is concatenated and passed to the 'representation block' to create the final features. 

        Args:
            x (torch.Tensor): _description_
            output (str, optional): _description_. Defaults to 'both'.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: _description_
        """
        # convert to the float data type and to the same device of the model if needed.
        x = x.to(torch.float).to(pu.get_module_device(self))
        return self._model.forward(x)

    def _forward_pass(self, batch, reduce_loss: bool = True):
        # extract the samples and the classes
        x, y = batch
        
        # forward pass
        model_output = self.forward(x)

        if model_output.ndim != 2 or model_output.shape[1] != self.num_classes:
            raise ValueError(f"the output is expected to be of the shape: {(model_output.shape[0], self.num_classes)}. Found: {model_output.shape}")

        # calculate the cross entropy loss
        loss_obj = F.cross_entropy(model_output, y, reduction=('mean' if reduce_loss else 'none'))

        # get the predictions
        predictions = torch.argmax(model_output, dim=1)
        
        # calculate the accuracy
        accuracy = torch.mean((predictions.to(torch.long) == y.to(torch.long)).to(torch.float32)).cpu().item()

        return model_output, predictions, loss_obj, accuracy

    def on_train_epoch_start(self) -> None:
        # this is carried out to avoid sharing values across epochs
        self.train_loss = 0
        self.train_acc = 0
        self.train_batch_count = 0

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        
        _, _, loss_obj, accuracy =  self._forward_pass(batch)

        self.log_dict({"train_cls_loss": loss_obj.cpu().item(),
                       "train_accuracy": round(accuracy, 5)})

        self.train_loss += loss_obj.cpu().item()
        self.train_acc += accuracy  
        self.train_batch_count += 1
        # make sure to return the loss object
        return loss_obj

    def on_train_epoch_end(self) -> None:
        # at the end of each epoch, average the results and save them
        # the total loss
        self.train_epoch_losses.append(self.train_loss / self.train_batch_count)
        # the train accuracy
        self.train_epoch_accs.append(self.train_acc / self.train_batch_count)
        
        # at the end of each epoch we will update the statistics of the accompanying swag model
        if hasattr(self, '_swag_model'):
            if self._swag_model is not None:
                if not isinstance(self._swag_model, SWAG):
                    raise TypeError(f"Found the 'swag_model' attribute in the wrapper, but of type {type(self._swag_model)} instead of {SWAG}")
                # proceed with updating the weights of the model
                self._swag_model.collect_model(base_model=self._model)

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
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the validation split in the source domain.")

        x, y = batch

        _, class_predictions, loss_obj, accuracy =  self._forward_pass(batch, reduce_loss=False)

        self.log_dict({"val_cls_loss": loss_obj.mean().cpu().item(),
                       "val_accuracy": round(accuracy, 5)}, 
                       add_dataloader_idx=False) # make sure that the dataloader index is not included in the name

        # make sure to track the validation losses
        self.val_loss += loss_obj.cpu().mean().item()
        self.val_acc += accuracy  
        self.val_batch_count += 1

        if batch_idx <= 2 and self.num_vis_images is not None:
            # extract the images with the largest loss
            top_losses, top_indices = torch.topk(input=loss_obj, k=self.num_vis_images, dim=-1)

            # convert the input images to numpy arrays
            b = x[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy()
            preds = class_predictions[top_indices].detach().cpu().numpy()
            labels = y[top_indices].detach().cpu().numpy()
            top_losses = top_losses.detach().cpu().numpy()

            data = [[wandb.Image(img), p, label, loss, self.current_epoch] 
                    for img, p, label, loss in zip(b, preds, labels, top_losses)]

            batch_df = pd.DataFrame(data=data, columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

            self.log_data = pd.concat([self.log_data, batch_df], axis=0)

            self.logger.log_table(key='val_summary', dataframe=self.log_data)

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
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the target domain.")

        _, y = batch
        _, _, target_loss_obj, target_accuracy = self._forward_pass(batch, reduce_loss=False)

        # the target accuracy is calculated as a mean over the batch (since batches are not necessarily of the same number of elements, we calculate the sum)
        # and divide by the total number of elements
        self.batch_target_accuracies.extend([target_accuracy for _ in range(len(y))])

        self.batch_target_losses.append(target_loss_obj.cpu())

    def on_validation_epoch_start(self) -> None:
        self.val_loss = 0
        self.val_acc = 0
        self.val_batch_count = 0

    def validation_step(self, 
                        batch: Union[List, Tuple], 
                        batch_idx: int, 
                        *args: Any, 
                        **kwargs: Any):        
        """
        This function handels both data from the validation split on the source domain or data from the target domain. 
        Depending on the format the method will call subsequent methods to handel each type of data accordingly. 
        
        The function expects the data to be passed in a CombinedLoader with 'sequential' mode assuring all data is passed 
        and without cycling

        Args:
            batch (Union[List[torch.Tensor, torch.Tensor, torch.Tensor], List[torch.Tensor, torch.Tensor]]): the data
            batch_idx (int): _description_

        Raises:
            TypeError: _description_
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

        if (len(self.batch_target_accuracies) + len(self.batch_target_losses) == 0):
            return
        
        # first convert the target losses field into a single tensor
        target_losses = torch.cat(self.batch_target_losses, dim=0)

        if len(target_losses) != len(self.batch_target_accuracies):
            raise ValueError(("Please make sure the number of elements in the 'self.target_losses' and the 'self.target_accuracies' are the same.\n"
                             f"\nFound: {len(target_losses)} losses and {len(self.batch_target_accuracies)} accuracies"))
        

        loss, accuracy = target_losses.mean().cpu().item(), np.mean(self.batch_target_accuracies)

        # log the results
        self.log_dict({"target_loss": loss, "target_accuracy": accuracy}, add_dataloader_idx=False)

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


def _set_dataloaders(train_dir: Union[str, Path], 
                     val_dir: Union[str, Path], 
                     target_dir: Union[str, Path], 
                     seed: int, 
                     image_transformation: tr,
                     batch_size: int,
                     num_workers: int,
                    ) -> Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
                               DataLoader[Tuple[torch.Tensor, torch.Tensor]], 
                               Optional[DataLoader[Tuple[torch.Tensor, torch.Tensor]]]]:
    
    # set the training dataloader
    train_ds = ImageFolder(root=train_dir, transform=image_transformation) 

    train_batch_size = min(len(train_ds) // 2 - 2, batch_size)
    train_batch_size += 1 if (train_batch_size % len(train_ds) == 1) else 0

    # specify the generators manually to make sure the exact same permutations are used for the same seed
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)

    train_dl = DataLoader(train_ds, 
                        batch_size=train_batch_size,
                        shuffle=True,
                        generator=train_gen, 
                        num_workers=num_workers,
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False)
                        )

    # create the validation dataloader
    val_ds = ImageFolder(root=val_dir, transform=image_transformation)

    target_ds, target_dl = None, None
    if target_dir is not None:
        # create the dataset
        target_ds = ImageFolder(root=target_dir, transform=image_transformation)

    
    val_batch_size = min(len(val_ds) // 2, batch_size)
    # increase the batch size until neither the number of items in the validation of target datasets are of modulo 1
    while True: 
        if len(val_ds) % val_batch_size == 1 or (target_ds is not None and len(target_ds) % val_batch_size == 1):
            val_batch_size += 1
        else:
            break


    val_gen = torch.Generator()
    val_gen.manual_seed(seed)
    val_dl = DataLoader(val_ds, 
                        batch_size=batch_size,
                        shuffle=False,
                        generator=val_gen,
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed),
                        persistent_workers=(True if num_workers > 0 else False)
                        )
    
    if target_dir is not None:
        target_gen = torch.Generator()
        target_gen.manual_seed(seed)

        target_dl = DataLoader(target_ds, 
                        batch_size=batch_size,
                        shuffle=False,
                        generator=target_gen,
                        num_workers=num_workers, 
                        worker_init_fn=partial(pu.set_worker_seed, seed=seed), 
                        persistent_workers=(True if num_workers > 0 else False)
                        )
        

    return train_dl, val_dl, target_dl


def train_classifier_wrapper(
                    classifier: ClassifierWrapper,
                    train_dir:Union[Path, str],
                    val_dir: Union[Path, str],
                    log_dir: Union[str, Path],
                    target_dir: Union[Path, str] = None,
                    run_name: str = None,
                    num_workers: int = 0,
                    batch_size: int = 1024,
                    num_epochs: int = 10, 
                    wandb_logger: WandbLogger = None, 
                    repeat_config: bool = False, 
                    seed: int = 69, 
                    use_wandb: bool = False # recently wandb has been acting up
                    ) -> ClassifierWrapper:

    pu.seed_everything(seed=seed)
    
    if use_wandb:
        wandb.init(project=WANDB_PROJECT_NAME, 
                name=run_name)

    if wandb_logger is None and use_wandb:
        # wandb details for logging
        
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


    # extract the transformation from the model    
    image_transformation = classifier._transform

    train_dl, val_dl, target_dl = _set_dataloaders(train_dir=train_dir, 
                                                   val_dir=val_dir, 
                                                   target_dir=target_dir,
                                                   image_transformation=image_transformation,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers, 
                                                   seed=seed)

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=1, 
                                        monitor="val_cls_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='classifier-{epoch:02d}-{val_cls_loss:06f}')

    # define the trainer
    log_step = 1 if len(train_dl) < 10 else 10
    
    try:
        trainer = L.Trainer(
                            accelerator='gpu',
                            devices=1,
                            logger=wandb_logger if use_wandb else False, # disable logging if we are not using wandb 
                            default_root_dir=log_dir,
                            max_epochs=num_epochs,
                            check_val_every_n_epoch=3,
                            log_every_n_steps=log_step,
                            callbacks=[checkpnt_callback],
                            deterministic=True)
    except:
        # the Adaptive Average pooling layer
        # does not have a deterministic implementation, hence the code above might raise an error since setting deterministic to True
        # is not very flexible (unlike the custom implementation)
        trainer = L.Trainer(
                            accelerator='gpu',
                            devices=1,
                            logger=wandb_logger if use_wandb else False,# disable logging if 'wandb' is not available 
                            default_root_dir=log_dir,
                            max_epochs=num_epochs,
                            check_val_every_n_epoch=3,
                            log_every_n_steps=log_step,
                            callbacks=[checkpnt_callback])

    # make sure to pass the correct form depending on the target directory 
    # val_dataloaders = ({_VALIDATION_DATALOADER_KEY: val_dl, _TARGET_DATALOADER_KEY: target_dl}) if target_dl is not None else (val_dl)
    val_dataloaders = (CombinedLoader([val_dl, target_dl], 
                                      mode='sequential')) if target_dl is not None else (val_dl)

    trainer.fit(model=classifier,
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
    if use_wandb:
        wandb.finish()

    print(f'Best checkpoint: {best_checkpoint}')

    # at this point we can access all the metrics
    metrics = {
            "train_losses": classifier.train_epoch_losses, 
            "train_accs": classifier.train_epoch_accs, 

            # for validation and target losses, the metrics of the sanity check are also included 
            # so we will discard them
            "val_losses": classifier.val_epoch_losses[1:], 
            "val_accs": classifier.val_epoch_accs[1:], 

            # make sure the code works both for the cases with and without target data [1:] will raise an error if the list is empty
            "target_losses": (classifier.target_epoch_losses[1:] if target_dir is not None else []),
            "target_accs": (classifier.target_epoch_accs[1:] if target_dir is not None else []),
            }

    # make sure to save the metrics after determining the best checkpoint (so the code above won't break)
    with open(os.path.join(log_dir, 'metrics.obj'), 'wb') as f: 
        pickle.dump(metrics, f)

    # the final step is to make sure to save the model to be later loaded with wrapper's checkpoint
    if classifier._swag_model is not None:
        torch.save({"model": classifier._swag_model.state_dict()}, os.path.join(log_dir, 'swag_checkpoint.pt'))

    return os.path.join(log_dir, best_checkpoint)


def classifier_predict(classifier: ClassifierWrapper, 
            data_dir: Union[str, Path],
            seed: int,
            batch_size: int = 1024, 
            num_workers: int = 2) -> float:
    pu.seed_everything(seed)
    # set the model to the 'eval' mode
    classifier.eval()

    dataset = ImageFolder(root=data_dir, transform=classifier._transform)

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=num_workers,
                            worker_init_fn=partial(pu.set_worker_seed, seed=seed)                            
                            )

    total_count = 0
    correct_count = 0

    with torch.no_grad():
         for x, y in tqdm(dataloader, desc='standard classifier: inference on the dataset'):
            # move the labels to the same device as the model's
            total_count += len(x)
            y = y.to(pu.get_module_device(classifier))
            cls_logits = classifier.forward(x)
            cls_predictions = torch.argmax(cls_logits, dim=1)
            correct_count += torch.sum((cls_predictions.to(torch.long) == y.to(torch.long)).to(torch.float)).cpu().item()
    
    accuracy = round(correct_count / total_count, 4)
    return accuracy

