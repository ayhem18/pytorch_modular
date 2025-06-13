"""
This script contains the main definition of the CBM loss
"""

import torch

from torch import nn
from typing import Union, Tuple

import torch.version


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
                reduce_loss: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        if not torch.all(torch.logical_or(input=(concepts_true == 1), other=(concepts_true == 0))):
            raise ValueError(f"the concept label should contain the values 1 or 0.")

        # for concepts predictions, the following criterion should be satisfied
        if concept_preds.shape != concepts_true.shape:
            raise ValueError((f"The concepts labels and concepts logits are expected to be of matching shapes\n"
                              f"Found logits: {concept_preds.shape}, labels: {concepts_true.shape}"))

        # the shape of the output depends on the 'reduce_loss' parameter
        # if reduce_loss is True, the output will be a scalar: the loss by each sample in the batch will be averaged
        # otherwise, all losses will be returned together

        reduction = ('mean' if reduce_loss else 'none')
        class_loss = nn.CrossEntropyLoss(reduction=reduction).forward(input=y_pred, target=y_true)
        concepts_loss = nn.BCEWithLogitsLoss(reduction=reduction).forward(input=concept_preds, target=concepts_true)

        # reduce_loss = True
        # class_loss of shape (1,)
        # concepts_loss of shape (1,)


        # reduce_loss = False
        # class_loss of shape (batch_size,) 
        # concepts_loss of shape (batch_size, num_concepts)

        # so if even when reduce_loss is True, the concepts_loss should be averaged across concepts
        concepts_loss = torch.mean(concepts_loss, dim=1) if reduction == 'none' else concepts_loss

        # the expression below works 
        final_loss = class_loss + self.alpha * concepts_loss

        if reduce_loss:
            assert final_loss.ndim == 0, f"The final loss should be a scalar when reduce_loss = True. Found: {final_loss.shape}"
        else:
            assert final_loss.shape == (concept_preds.shape[0],), f"The final loss should be a vector when reduce_loss = False. Found: {final_loss.shape}"

        if return_all:
            return concepts_loss, class_loss, final_loss

        return final_loss
