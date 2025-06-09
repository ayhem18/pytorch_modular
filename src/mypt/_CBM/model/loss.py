"""
This script contains the main definition of the CBM loss
"""

import torch

from torch import nn
from typing import Union, Tuple


class CBMLoss(nn.Module):
    """
    This Loss class will be used when the concept labels can be seed as probability distribution. 
    In other words, the label values all sum up to '1'.        
    """

    def __init__(self,
                 alpha: float,
                 *args, 
                 **kwargs,):
        # call the super class constructor
        super().__init__(*args, **kwargs)
        # alpha is the balancing parameter
        self.alpha = alpha

    def forward(self,
                concept_preds: torch.Tensor,
                concepts_true: torch.Tensor,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                return_all: bool = False, 
                reduce_loss: bool = True) -> Union[nn.Module, Tuple[nn.Module, nn.Module, nn.Module]]:

        # for concepts predictions, the following criterion should be satisfied
        if concept_preds.shape != concepts_true.shape:
            raise ValueError((f"The concepts labels and concepts logits are expected to be of matching shapes\n"
                              f"Found logits: {concept_preds.shape}, labels: {concepts_true.shape}"))
        
        # the shape of the output graetly depends on 
        reduction = ('mean' if reduce_loss else 'none')
        class_loss = nn.CrossEntropyLoss(reduction=reduction)(input=y_pred, target=y_true)
        # set the reduction parameter in the 'loss' associated with concepts
        concepts_loss_object = nn.CrossEntropyLoss(reduction=reduction)
        concept_loss = concepts_loss_object(concept_preds, concepts_true)
        final_loss = class_loss + self.alpha * concept_loss

        if return_all:
            return concept_loss, class_loss, final_loss

        return final_loss

class BinaryCBMLoss(nn.Module):
    """
        Concept labels were initally consider as the distribution of the similarities between a given an image and the predetermined set of concepts.
        Nevertheless, it is also possible for the concept labels to represent binary values: basically whether a certain concept is present in a given image or not.
        This requires using Binarly Cross Entropy losses
    """
    def __init__(self,
                 alpha: float,
                 *args, 
                 **kwargs,):
        # call the super class constructor
        super().__init__(*args, **kwargs)
        # alpha is the balancing parameter
        self.alpha = alpha

    def forward(self,
                concept_preds: torch.Tensor,
                concepts_true: torch.Tensor,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                return_all: bool = False, 
                reduce_loss: bool = True) -> Union[nn.Module, Tuple[nn.Module, nn.Module, nn.Module]]:

        if not torch.all(torch.logical_or(input=(concepts_true == 1), other=(concepts_true == 0))):
            raise ValueError(f"the concept label should contain the values 1 or 0.")

        # for concepts predictions, the following criterion should be satisfied
        if concept_preds.shape != concepts_true.shape:
            raise ValueError((f"The concepts labels and concepts logits are expected to be of matching shapes\n"
                              f"Found logits: {concept_preds.shape}, labels: {concepts_true.shape}"))

        # the shape of the output greatly depends on 
        reduction = ('mean' if reduce_loss else 'none')
        class_loss = nn.CrossEntropyLoss(reduction=reduction)(input=y_pred, target=y_true)
        
        # set the reduction parameter in the 'loss' associated with concepts
        concepts_loss_object = nn.BCEWithLogitsLoss(reduction=reduction)
        concept_loss = concepts_loss_object(concept_preds, concepts_true)

        # if reduction is set to 'none' we will compute the mean at each element of the batch
        concept_loss = torch.mean(concept_loss, dim=1) if reduction == 'none' else concept_loss

        final_loss = class_loss + self.alpha * concept_loss
        # final_loss = concept_loss

        if return_all:
            return concept_loss, class_loss, final_loss

        return final_loss
