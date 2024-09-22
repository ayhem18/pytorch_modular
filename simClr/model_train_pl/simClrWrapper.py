"""
This file contains a pytorch lightning wrapper for the SimClr Model
"""

import torch, warnings

from typing import Union, Tuple, List, Dict, Optional

from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule

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
    def _set_learning_rates(cls, lrs: Union[(List | Tuple)[float, float], float]) -> Tuple[float, float]:
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
                # loss arguments
                temperature: float,
                debug_loss:bool,
                # optimizer arguments
                lrs: Union[Tuple[float], float], 
                num_epochs: int,
                num_warmup_epochs:int,                 

                ):
        
        # the model
        self.model: SimClrModel = None
        self.input_shape = input_shape

        # the loss function
        self.loss_obj = SimClrLoss(temperature=temperature, debug=debug_loss)

        # the optimizer parameters
        self.lr1, self.lr2 = self._set_learning_rates(lrs)
        self._num_epochs = num_epochs
        self._num_warmup_epochs = num_warmup_epochs

        # logging parameters
        self.log_per_batch = log_per_batch


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(x)


    def _batch_log(self, log_dict: Dict, batch_idx: int) -> None:
        """
        Args:
            log_dict (Dict): the batch output saved as a dictionary
            batch_idx (int): The index of the batch (might be used to decide which batch output to log)
        """
        if batch_idx % self.log_per_batch == 0:
            pass
        pass
    

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x1_batch, x2_batch = batch
        
        # concatenate the input 
        x = torch.cat([x1_batch, x2_batch])
        
        # model forward
        _, g_x = self.forward(x)
        
        # calculate the loss
        batch_loss_obj = self.loss_obj.forward(g_x)

        log_dict = {"batch_train_loss": batch_loss_obj.item()}

        if isinstance(batch_loss_obj, Tuple):
            batch_loss_obj, positive_pairs_sims, negative_pairs_sims = batch_loss_obj            

            pos_neg_sim_stats = {
                        "train_avg_positive_pair_sim": torch.mean(positive_pairs_sims).item(),
                        "train_avg_negative_pair_sim": torch.mean(negative_pairs_sims).item()
                    }

            log_dict.update(pos_neg_sim_stats)

        return batch_loss_obj, log_dict

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> SimClrLoss:
        batch_loss_obj, log_dict = self._step(batch)
        self._batch_log(log_dict=log_dict, batch_idx=batch_idx)
        return batch_loss_obj


    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_index:int) -> torch.Tensor | Dict[str, torch.Any] | None:
        batch_loss_obj, log_dict = self._step(val_batch)
        self._batch_log(log_dict=log_dict, batch_idx=0) # log all validation batches
        return batch_loss_obj
        

    def configure_optimizers(self):
        # importing the Lars optimizer inside this function, simply because the "flash" package is noticeably slow to load ....
        # hence slowing down the entire code base even when I am not training
        from flash.core.optimizers import LARS

        optimizer = LARS(params=[{"params": self.model.fe.parameters(), "lr": self.lr1}, # using different learning rates with different components of the model 
                                {"params": self.model.flatten_layer.parameters(), "lr": self.lr1},
                                {"params": self.model.ph.parameters(), "lr": self.lr2}
                                ], 
                                lr=self.lr1, 
                                weight_decay=10 ** -6)
                    
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self._num_epochs - self._num_warmup_epochs)

        if self._num_epochs > 0:
            optimizer, lr_scheduler = set_warmup_epochs(optimizer=optimizer, 
                                                        main_lr_scheduler=cosine_scheduler, 
                                                        num_warmup_epochs=self._num_warmup_epochs, 
                                                        warmup_lr_scheduler='linear')

            return optimizer, lr_scheduler

        return optimizer, cosine_scheduler 


class ResnetSimClrWrapper(SimClrModelWrapper):
    def __init__(self,
                input_shape: Tuple[int, int, int],
                output_dim: int,
                num_fc_layers: int,

                #logging parameters
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
        