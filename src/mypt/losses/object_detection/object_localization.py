"""
This script contain an implementation of the main losses used in the context of Object Detection
"""

import torch
from typing import Union, Dict


class ObjectLocalizationLoss(torch.nn.Module):
    _l1_name = "obj_indicator_loss"
    _l2_name = "bbox_loss"
    _l3_name = "cls_loss"
    _loss_name = "final_loss"

    def __init__(self, 
                reduction: str = 'none',
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, 
                y: torch.Tensor, 
                all: bool=False) -> Union[torch.Tensor, Dict]:
        # let's first check a couple of things: 
        if x.ndim != 2:
            raise ValueError(f"The current implementation only accepts 2 dimensional input. Found: {x.ndim} -dimensional input.")

        if tuple(x.shape) != tuple(y.shape):
            raise ValueError(f"Object localization expects the prediction and the label to be of the same shape. Founnd: x as {x.shape} and y as {y.shape}")

        obj_indicator_pred, bounding_boxes_pred, classification_pred = x[:, [0]], x[:, 1:5], x[:, 5:] 
        y_obj, y_bbox, y_cls = y[:, [0]], y[:, 1:5], y[:, 5:]

        y_index = y[:, 0].to(torch.bool)

        l1 = torch.nn.BCEWithLogitsLoss(reduction=self.reduction).forward(obj_indicator_pred, y_obj)
        # the loss on the bounding boxes will be the sum of mse on the four corners
        l2 = torch.sum(torch.nn.MSELoss(reduction=self.reduction).forward(bounding_boxes_pred[y_index], y_bbox[y_index]), dim=1, keepdim=True) 
        l3 = torch.nn.CrossEntropyLoss(reduction=self.reduction).forward(classification_pred[y_index], y_cls[y_index]).unsqueeze(dim=1)

        for l in [l1, l2, l3]:
            if torch.any(torch.logical_or(l.isnan(), l.isinf())): 
                raise ValueError(f"The loss has reached infinity of nan values !!!")

        # in the very rare scenario, where all samples are background (a.k.a: y_index will have all zeros)
        if torch.sum(y_index) == 0:
            l1, l2, l3 = torch.mean(l1, dim=0), torch.Tensor(0, dtype=torch.float32), torch.Tensor(0, dtype=torch.float32)
            final_loss = l1 # only include the l1 loss in this case
        
        else:
            # since l1 contains all samples (samples with background and object of interest)
            # while l2 and l3 have been indexed with rows of only with interest
            # l1, l2, l3 do not necessarily share the same shape: the final loss can only be calculated as an average
            l1, l2, l3 = torch.mean(l1, dim=0), torch.mean(l2, dim=0), torch.mean(l3, dim=0)
            final_loss = l1 + l2 + l3
        
        if all:
            return {self._loss_name: final_loss, 
                    self._l1_name: l1, 
                    self._l2_name: l2, 
                    self._l3_name: l3}

        return final_loss
