"""
This file contains a pytorch lightning wrapper for the SimClr Model
"""

import torch, warnings, torch_optimizer as topt, numpy as np

from typing import Union, Tuple, List, Dict, Optional, Iterator
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
from clearml import Task, Logger

from mypt.losses.simClrLoss import SimClrLoss
from mypt.code_utilities import pytorch_utilities as pu, directories_and_files as dirf
from mypt.models.simClr.simClrModel import SimClrModel, ResnetSimClr
from mypt.schedulers.warmup import set_warmup_epochs



class SimClrModelWrapper(LightningModule):
    @classmethod
    def verify_optimizer_parameters(cls, num_epochs: int, num_warmup_epochs:int):
        if num_warmup_epochs >= num_epochs:
            raise ValueError(f"The number of warmup epochs must be strictly less than the total number of epochs !!. found warmup : {num_warmup_epochs}, and epochs: {num_epochs}")

        if num_warmup_epochs == 0:
            warnings.warn(message="The number of warmup epochs is set to 0 !!")

    @classmethod
    def _set_learning_rates(cls, lrs: Union[Tuple[float, float], float]) -> Tuple[float, float]:
        if isinstance(lrs, float):
            lr1, lr2 = lrs, 10 * lrs
        elif isinstance(lrs, (Tuple,List)) and len(lrs) == 2:
            lr1, lr2 = lrs

        elif isinstance(lrs, (Tuple,List)) and len(lrs) == 1:
            lr1, lr2 = lrs[0], lrs[0] * 10
            
        else:
            raise ValueError(f"The current implementation supports at most 2 learning rates. Found: {len(lrs)} learning rates")

        return lr1, lr2
    

    def __init__(self,
                input_shape: Tuple[int, int, int],

                #logging arguments
                log_per_batch: int,
                logger: Optional[Logger], 
                # loss arguments
                temperature: float,
                debug_loss:bool,
                # optimizer arguments
                lrs: Union[Tuple[float], float], 
                num_epochs: int,
                num_warmup_epochs:int,                 

                ):
        # super class call
        super().__init__()

        # the model
        self.model: SimClrModel = None
        self.input_shape = input_shape

        # make sure that the loss override the nn.Module and actually calls the super constructor
        self._loss = SimClrLoss(temperature=temperature, debug=debug_loss)
        
        # the optimizer parameters
        self.lr1, self.lr2 = self._set_learning_rates(lrs)
        self._num_epochs = num_epochs
        self._num_warmup_epochs = num_warmup_epochs

        # logging parameters: 
        # this parameter has to be passed since, I will doing the logging to clearml myself 
        self.log_per_batch = log_per_batch  
        self.myLogger = logger

        # logging variables
        self.train_batch_logs: List[Dict[str, float]] = [] # a field to save the metrics logged during the training batches
        self.val_batch_logs: List[Dict[str, float]] = [] # a field to save the metrics logged during the validation batches


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x)


    def _batch_log(self, 
                   log_dict: Dict, 
                   batch_idx: int) -> None:
        """
        Args:
            log_dict (Dict): the batch output saved as a dictionary
            batch_idx (int): The index of the batch (might be used to decide which batch output to log)
        """

        # make sure to refer to myLogger and not self.logger
        if self.myLogger is None or batch_idx % self.log_per_batch != 0: 
            return
        
        # the flexibility of clearML comes with the issue of specifying the iteration number
        # but there is no way Pytorch lightning is not keeping track of the global step...

        for key, value in log_dict.items():
            self.myLogger.report_scalar(title=key, 
                                    series=key, 
                                    value=value, 
                                    iteration=self.global_step # 
                                    )
            

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], split: str) -> Tuple[torch.Tensor, Dict]:
        x1_batch, x2_batch = batch
        
        # concatenate the input 
        x = torch.cat([x1_batch, x2_batch])
        
        # model forward
        _, g_x = self.forward(x)
        
        # calculate the loss
        batch_loss_obj = self._loss.forward(g_x)

        log_dict = {}

        if isinstance(batch_loss_obj, Tuple):
            batch_loss_obj, positive_pairs_sims, negative_pairs_sims = batch_loss_obj            

            pos_neg_sim_stats = {
                        f"{split}_avg_positive_pair_sim": torch.mean(positive_pairs_sims).item(),
                        f"{split}_avg_negative_pair_sim": torch.mean(negative_pairs_sims).item()
                    }

            log_dict.update(pos_neg_sim_stats)

        # at this point of the code we know that batch_loss_obj actually represents the loss tensor and the tuple with the logging stats
        log_dict.update({f"batch_{split}_loss": batch_loss_obj.item()})

        return batch_loss_obj, log_dict

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> SimClrLoss:
        batch_loss_obj, log_dict = self._step(batch, split='train')
        self._batch_log(log_dict=log_dict, batch_idx=batch_idx)
        
        # save only the training loss
        self.train_batch_logs.append(log_dict['batch_train_loss'])
        return batch_loss_obj


    def on_train_epoch_end(self):
        # calculate the average of batch losses
        train_epoch_loss = np.mean(self.train_batch_logs)

        # log to ClearMl
        if self.myLogger is not None:
            self.myLogger.report_scalar(
                                        title="train_epoch_loss", 
                                        series="train_epoch_loss", 
                                        value=train_epoch_loss, # 
                                        iteration=int(round(self.global_step / len(self.train_batch_logs))) # the epoch index can be deduced from global step and the length of self.train_batch_logs
                                        )
            
        # clear the list
        self.train_batch_logs.clear()

        # log the epoch validation loss to use it for checkpointing
        self.log(name='train_epoch_loss', value=train_epoch_loss)


    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_index:int) -> SimClrLoss:
        batch_loss_obj, log_dict = self._step(val_batch, split='val')
        self._batch_log(log_dict=log_dict, batch_idx=0) # log all validation batches
            
        # track the validation batch loss
        self.val_batch_logs.append(log_dict['batch_val_loss'])
        return batch_loss_obj


    # make sure to name the methods according to the ModelHooks mixin extended by the LightningModule
    def on_validation_epoch_end(self):
        # calculate the average of batch losses
        val_epoch_loss = float(np.mean(self.val_batch_logs))

        # log to ClearMl
        if self.myLogger is not None:
            self.myLogger.report_scalar(
                                        title="val_epoch_loss", 
                                        series="val_epoch_loss", 
                                        value=val_epoch_loss, # 
                                        iteration=int(round(self.global_step / len(self.train_batch_logs)))
                                        )
        # clear the list
        self.val_batch_logs.clear()

        # log the epoch validation loss to use it for checkpointing
        self.log(name='val_epoch_loss', value=val_epoch_loss)


    def configure_optimizers(self):
        # I found only one good implementation of LARS through the lightning.flash package. The latter requires pytorch lightning version < 2.0 as a dependency. The current version is 2.4 ...
        # switching to a supported implementation of Lamb (another large batch learning scheduler)

        optimizer = topt.Lamb(params=[{"params": self.model.fe.parameters(), "lr": self.lr1}, # using different learning rates with different components of the model 
                                {"params": self.model.flatten_layer.parameters(), "lr": self.lr1},
                                {"params": self.model.ph.parameters(), "lr": self.lr2}
                                ],
                                lr=self.lr1,
                                weight_decay=10 ** -6
                                )
                    
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self._num_epochs - self._num_warmup_epochs)

        if self._num_epochs > 0:
            lr_scheduler_with_warmup = set_warmup_epochs(optimizer=optimizer, 
                                                        main_lr_scheduler=cosine_scheduler, 
                                                        num_warmup_epochs=self._num_warmup_epochs, 
                                                        warmup_lr_scheduler='linear')

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_with_warmup}

        return {"optimizer": optimizer, "lr_scheduler": cosine_scheduler}


    # added to ensure that the model behaves as expected: more related to Pytorch  nn.Module than Pytorch Lightning
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self


    def children(self) -> Iterator[torch.nn.Module]:
        # overloading this method to return the correct children of the wrapper: those of the self.model field
        return self.model.children()


    def modules(self) -> Iterator[torch.nn.Module]:
        return self.model.modules()


    def named_children(self) -> Iterator[Tuple[str, torch.nn.Module]]:
        return self.model.named_children()
        

class ResnetSimClrWrapper(SimClrModelWrapper):
    def __init__(self,
                input_shape: Tuple[int, int, int],
                output_dim: int,
                num_fc_layers: int,

                #logging parameters
                logger: Optional[Logger],
                log_per_batch: int,
                # loss parameters
                temperature: float,
                debug_loss:bool,
                # optimizer parameters
                lrs: Union[Tuple[float], float], 
                num_epochs: int,
                num_warmup_epochs:int,

                # more model arguments
                dropout: Optional[float] = None,
                architecture: int = 50, 
                freeze: Union[int, bool]=False,  
                ):

        super().__init__(
                input_shape=input_shape,
                #logging parameters
                log_per_batch=log_per_batch,
                logger=logger,
                # loss parameters
                temperature=temperature,
                debug_loss=debug_loss,
                # optimizer parameters
                lrs=lrs, 
                num_epochs=num_epochs,
                num_warmup_epochs=num_warmup_epochs)

        self.model = ResnetSimClr(
                                input_shape=self.input_shape, 
                                output_dim=output_dim, 
                                num_fc_layers=num_fc_layers, 
                                
                                dropout=dropout, 
                                freeze=freeze, 
                                architecture=architecture
                                )
        