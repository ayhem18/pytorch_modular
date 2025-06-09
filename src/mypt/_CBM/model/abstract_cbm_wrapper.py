"""
This script defines a Pytorch Lightning wrapper around Concept Bottleneck models as well as training, and predicting functionalities. 
"""

import os, torch, wandb, pickle

import pytorch_lightning as L
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Any, Union, Tuple, Dict, List, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from functools import partial
from tqdm import tqdm
from collections import defaultdict

from ..data.concept_dataloaders import set_dataloaders
from ..data.datasets.conceptDataset import ConceptDataset
from ...code_utilities import directories_and_files as dirf
from ...code_utilities import pytorch_utilities as pu
from ...swag.posteriors.swag import SWAG

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WANDB_PROJECT_NAME = 'CBM-UDA'

class CbmWrapper(L.LightningModule):
    def __init__(self, 
                input_shape: Tuple,
                num_concepts: int, 
                num_classes: int,   
                optimizer_class: callable,
                loss: callable,
                learning_rate: Optional[float] = 10 ** -4,
                loss_coefficient: float = 0.5,                 
                num_vis_images: int = 3,
                scheduler_class: Optional[callable] = None,
                optimizer_keyargs: Optional[Dict] = None,
                scheduler_keyargs: Optional[Dict] = None,   
                ):
        super().__init__()
        
        self.input_shape = input_shape 
        self.num_concepts = num_concepts
        self.num_classes = num_classes        

        self.lr = (learning_rate if learning_rate is not None else 10 ** -4)
        self.opt = optimizer_class
        self.opt_args = optimizer_keyargs

        # save the scheduler arguments
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_keyargs

        # create the loss object
        self._loss = loss(alpha=(loss_coefficient if loss_coefficient is not None else 0.5))

        # save the number of imgaes to be logged at each validation step
        self.num_vis_images = num_vis_images

        # the concrete class has to implement the fields below
        self._model, self._transform, self._swag_model = None, None, None

        # create a field that will save all the data to be logged
        self.log_data = pd.DataFrame(data=[], columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

        self.save_hyperparameters()

        # create some fields to save the model's metrics: total loss, classification loss, concept loss, accuracy
        self.train_epoch_losses = []
        self.train_epoch_cls_losses = []
        self.train_epoch_concept_losses = []
        self.train_epoch_accs = []

        # the same for validation losses
        self.val_epoch_losses = []
        self.val_epoch_cls_losses = []
        self.val_epoch_concept_losses = []
        self.val_epoch_accs = []

        # add some fields to track the performance of the model on the target domain: this fields are used mainly
        # to account for the uneven split in batches when calculating the target accuracy      
        self.batch_target_accuracies = []
        self.batch_target_losses = []

        # these fields are used mainly to save the losses after each validation epoch (if the target dataloader is passed of course)
        self.target_epoch_losses = []
        self.target_epoch_accs = []

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
        # convert to the float data type
        x = x.to(torch.float).to(pu.get_module_device(self))
        return self._model.forward(x)

    def _forward_pass(self, batch, reduce_loss: bool = True):
        # first seperate the input
        x, concept_labels, class_labels = batch
        
        # forward pass through the model
        concept_logits, class_logits = self.forward(x)

        # calculate the loss
        concept_loss, class_loss, final_loss = self._loss.forward(concept_preds=concept_logits, 
                           concepts_true=concept_labels,
                           y_pred=class_logits,
                           y_true=class_labels, 
                           return_all=True, 
                           reduce_loss=reduce_loss)
        # calculate accuracy
        class_predictions = torch.argmax(class_logits, dim=1)

        accuracy = torch.mean((class_predictions.to(torch.long) == class_labels.to(torch.long)).to(torch.float)).cpu().item()

        return concept_loss, class_loss, final_loss, class_predictions, accuracy

    def on_train_epoch_start(self) -> None:
        # this is carried out to avoid sharing values across epochs
        self.train_loss = 0
        self.train_acc = 0
        self.train_concept_loss = 0
        self.train_cls_loss = 0
        self.train_batch_count = 0
        
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        concept_loss, class_loss, final_loss, _, accuracy =  self._forward_pass(batch)

        self.log_dict({"train_cls_loss": class_loss.cpu().item(),
                       "train_concept_loss": concept_loss.cpu().item(),
                       "train_loss": final_loss.cpu().item(), 
                       "train_accuracy": round(accuracy, 5)})
        
        # make sure to add each of the metrics to the variable that is supposed to store it
        self.train_loss += final_loss.cpu().item()
        self.train_concept_loss += concept_loss.cpu().item()
        self.train_cls_loss += class_loss.cpu().item()
        self.train_acc += accuracy  
        self.train_batch_count += 1
        return final_loss 
    
    def on_train_epoch_end(self) -> None:
        # at the end of each epoch, average the results and save them
        # the total loss
        self.train_epoch_losses.append(self.train_loss / self.train_batch_count)
        # the classification loss
        self.train_epoch_cls_losses.append(self.train_cls_loss / self.train_batch_count)
        # the concept losses
        self.train_epoch_concept_losses.append(self.train_concept_loss / self.train_batch_count)
        # the train accuracy
        self.train_epoch_accs.append(self.train_acc / self.train_batch_count)

        # at the end of each epoch we will update the statistics of the accompanying swag model
        if hasattr(self, '_swag_model'):
            if self._swag_model is not None:
                if not isinstance(self._swag_model, SWAG):
                    raise TypeError(f"Found the 'swag_model' attribute in the wrapper, but of type {type(self._swag_model)} instead of {SWAG}")
                # updates the weights of the swag model
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
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 3 and 
                isinstance(batch[0], torch.Tensor) and 
                isinstance(batch[1], torch.Tensor) and
                isinstance(batch[2], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the validation split in the source domain.")

        x, _, class_labels = batch
        concept_loss, class_loss, final_loss, class_predictions, accuracy =  self._forward_pass(batch, reduce_loss=False)

        self.log_dict({"val_cls_loss": class_loss.cpu().mean().item(),
                       "val_concept_loss": concept_loss.cpu().mean().item(),
                       "val_loss": final_loss.cpu().mean().item(), 
                       "val_accuracy": round(accuracy, 5)},
                        add_dataloader_idx=False)

        # make sure to track the validation losses
        self.val_loss += final_loss.cpu().mean().item()
        self.val_concept_loss += concept_loss.cpu().mean().item()
        self.val_cls_loss += class_loss.cpu().mean().item()
        self.val_acc += accuracy  
        self.val_batch_count += 1

        if batch_idx <= 2 and self.num_vis_images is not None:
            # extract the images with the largest loss
            top_losses, top_indices = torch.topk(input=class_loss, k=self.num_vis_images, dim=-1)

            # convert the input images to numpy arrays
            b = x[top_indices].detach().cpu().permute(0, 2, 3, 1).numpy()
            preds = class_predictions[top_indices].detach().cpu().numpy()
            labels = class_labels[top_indices].detach().cpu().numpy()
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
        if not (isinstance(batch, (Tuple, List)) and len(batch) == 3 
                and isinstance(batch[0], torch.Tensor) 
                and isinstance(batch[1], torch.Tensor) 
                and isinstance(batch[2], torch.Tensor)):
            raise ValueError(f"Make sure to pass the data from the target domain.")

        x, _, y = batch        
        _, target_cls_loss, _, _, target_accuracy = self._forward_pass(batch, reduce_loss=False)

        # the target accuracy is calculated as a mean over the batch (since batches are not necessarily of the same number of elements, we calculate the sum)
        # and divide by the total number of elements
        self.batch_target_accuracies.extend([target_accuracy for _ in range(len(y))])
        self.batch_target_losses.append(target_cls_loss.cpu())

    def on_validation_epoch_start(self) -> None:
        self.val_loss = 0
        self.val_acc = 0
        self.val_concept_loss = 0
        self.val_cls_loss = 0
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
        # the classification loss
        self.val_epoch_cls_losses.append(self.val_cls_loss / self.val_batch_count)
        # the concept loss
        self.val_epoch_concept_losses.append(self.val_concept_loss / self.val_batch_count)
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
    

def train_cbm_wrapper(cbm_wrapper: CbmWrapper, 
                    concepts: Union[Dict[str, List[str]], List[str]], 
                    train_dir:Union[Path, str],
                    val_dir: Union[Path, str],
                    representation: int,
                    similarity: str,
                    log_dir: Union[str, Path],
                    target_dir: Union[str, Path] = None,
                    run_name: str = None,
                    batch_size: int = 512,
                    num_epochs: int = 10,
                    wandb_logger: WandbLogger = None,
                    repeat_config: bool = False,
                    remove_existing: bool = False, 
                    seed: int = 69, 
                    num_workers: int = 0,
                    use_wandb:bool = False,
                    **kwargs) -> Union[str, Path]:

    pu.seed_everything(seed=seed)

    if use_wandb:
        wandb.init(project=WANDB_PROJECT_NAME, 
                name=run_name)

    if use_wandb and wandb_logger is None:
        wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                                log_model=False, 
                                save_dir=log_dir, 
                                name=run_name)

    train_dir = dirf.process_path(train_dir, file_ok=False,)
    val_dir = dirf.process_path(val_dir, file_ok=False,)
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

    train_dl, val_dl, target_dl = set_dataloaders(train_dir=train_dir,
                                                  val_dir=val_dir,
                                                  target_dir=target_dir,
                                                  representation=representation,
                                                  similarity=similarity,
                                                  concepts=concepts,
                                                  image_transformation=cbm_wrapper._transform,
                                                  remove_existing=remove_existing,
                                                  seed=seed,
                                                  kwargs=kwargs, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers, 
                                                  **kwargs)

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=1, 
                                        monitor="val_loss", # the best checkpoint is the one with the least overall loss on the validation split
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='CBM-{epoch:02d}-{val_loss:06f}')


    log_step = 1 if len(train_dl) < 10 else 10

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu',
                        devices=1,
                        logger=wandb_logger if use_wandb else False, # disable logging if we are not using wandb 
                        default_root_dir=log_dir,
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=3,
                        log_every_n_steps=log_step,
                        callbacks=[checkpnt_callback])

    val_dataloaders = (CombinedLoader([val_dl, target_dl], mode='sequential')) if target_dl is not None else (val_dl)

    trainer.fit(model=cbm_wrapper,
                train_dataloaders=train_dl,
                val_dataloaders=val_dataloaders
                )

    # at this point we can access all the metrics
    metrics = {
            "train_losses": cbm_wrapper.train_epoch_losses, 
            "train_cls_losses": cbm_wrapper.train_epoch_cls_losses, 
            "train_concept_losses": cbm_wrapper.train_epoch_concept_losses,
            "train_accs": cbm_wrapper.train_epoch_accs, 

            # for validation and target losses, the metrics of the sanity check are also included 
            # so we will discard them
            "val_losses": cbm_wrapper.val_epoch_losses[1:], 
            "val_cls_losses": cbm_wrapper.val_epoch_cls_losses[1:], 
            "val_concept_losses": cbm_wrapper.val_epoch_concept_losses[1:],
            "val_accs": cbm_wrapper.val_epoch_accs[1:], 

            # make sure the code works both for the cases with and without target data [1:] will raise an error if the list is empty
            "target_losses": (cbm_wrapper.target_epoch_losses[1:] if target_dir is not None else []),
            "target_accs": (cbm_wrapper.target_epoch_accs[1:] if target_dir is not None else []),
            }
    
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
    
    if use_wandb:
        wandb.finish()

    print(f'Best checkpoint: {best_checkpoint}')

    # make sure to save the metrics after determining the best checkpoint (so the code above won't break)
    with open(os.path.join(log_dir, 'metrics.obj'), 'wb') as f: 
        pickle.dump(metrics, f)

    # the final step is to save the swag model to be later loaded with wrapper's checkpoint
    if cbm_wrapper._swag_model is not None:
        torch.save({"model": cbm_wrapper._swag_model.state_dict()}, os.path.join(log_dir, 'swag_checkpoint.pt'))

    return os.path.join(log_dir, best_checkpoint)

def cbm_predict(cbm_wrapper: CbmWrapper, 
            data_dir: Union[str, Path],
            concepts: Union[List[str], Dict[str, List[str]]], 
            seed: int,
            batch_size: int = 1024,
            num_workers: int = 3, 
            ) -> float:
    
    pu.seed_everything(seed)
    # set the model to the 'eval' mode
    cbm_wrapper.eval()
    
    # the data directly might contain concept labels which are not image data. Hence we cannot use the default ImageFolder dataset
    # provided by Pytorch. Since we are using the concepts labels in the inference, we can use any Concept dataset
    dataset = ConceptDataset(root=data_dir,
                             concepts=concepts,
                             image_transform=cbm_wrapper._transform, 
                             remove_existing=False, 
                             label_generation_batch_size=batch_size       
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
                            worker_init_fn=partial(pu.set_worker_seed, seed=seed)
                            )
    total_count = 0
    correct_count = 0

    with torch.no_grad():
        for x, _, cls_labels in tqdm(dataloader, desc=f'CBM: inference on the dataset'):
            # move the labels to the same device as the model's
            cls_labels = cls_labels.to(pu.get_module_device(cbm_wrapper))
            total_count += len(x)
            _, cls_logits = cbm_wrapper.forward(x)
            cls_predictions = torch.argmax(cls_logits, dim=1)
            correct_count += torch.sum((cls_predictions.to(torch.long) == cls_labels.to(torch.long)).to(torch.float)).cpu().item()
    
    accuracy = round(correct_count / total_count, 4)
    return accuracy

# let's define a function to calculate the average concepts loss per class
def losses_per_cls(wrapper: CbmWrapper, 
                   concepts_dataloader: DataLoader, 
                   concepts_loss_function: torch.nn.Module) -> Tuple[Dict[int, float]]:
    """
    This function computes the average concept and classification loss per class for a given CBM
    """
    
    device = pu.get_default_device()
    wrapper.to(device)
    concepts_loss_per_cls = defaultdict(lambda : 0)
    cls_loss_per_cls = defaultdict(lambda : 0)
    count_per_cls = defaultdict(lambda : 0)

    wrapper.eval()


    with torch.no_grad():
        # iterate through the dataloader
        for x, y_c, y in tqdm(concepts_dataloader, desc=f'calculating losses per class'):
            # set everything to the correct device
            x = x.to(device)
            y_c = y_c.to(device)
            y = y.to(device)
            
            # extract the classes present in the batch
            batch_classes = torch.unique(y).cpu().tolist()
            # iterate through the different classes
            for label in batch_classes:
                # for each class extract the samples belonging to the given class and the corresponding labels
                mask = (y == label)
                x_label = x[mask]
                y_c_label = y_c[mask]
                y_label = y[mask]

                if len(x_label) == 1:
                    continue

                # save the number of samples belonging 
                count_per_cls[label] += len(x_label)
                
                # forward pass
                concept_logits, cls_logits = wrapper.forward(x_label)

                # calculate the classification loss: sum up the losses
                cls_loss_per_cls[label] += torch.sum(torch.nn.CrossEntropyLoss(reduction='none').forward(cls_logits, y_label)).cpu().item()
                
                # calculate the concepts_loss
                if hasattr(concepts_loss_function, 'reduction'):
                    concepts_loss_function.reduction = 'none'
                
                # calculate the concepts loss: sum up the losses
                concepts_loss_per_cls[label] += torch.sum(torch.mean(concepts_loss_function.forward(concept_logits, y_c_label), dim=1)).cpu().item()


        # at the very end, average by the number of samples
        for label, label_count in count_per_cls.items():
            cls_loss_per_cls[label] /= label_count
            concepts_loss_per_cls[label] /= label_count

        return concepts_loss_per_cls, cls_loss_per_cls
