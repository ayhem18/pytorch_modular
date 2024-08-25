"""
This script contain an implementation of the main losses used in the context of Object Detection
"""

import torch


class ObjectLocalizationLoss(torch.nn.Module):
    def __init__(self, 
                compact: bool,
                reduction: str = 'none',
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # let's first check a couple of things: 
        if x.ndim != 2:
            raise ValueError(f"The current implementation only accepts 2 dimensional input. Found: {x.ndim} -dimensional input.")

        if tuple(x.shape) != tuple(y.shape):
            raise ValueError(f"Object localization expects the prediction and the label to be of the same shape. Founnd: x as {x.shape} and y as {y.shape}")

        obj_indicator_pred, bounding_boxes_pred, classification_pred = x[:, [0]], x[:, 1:5], x[:, 5:] 
        y_obj, y_bbox, y_cls = y[:, [0]], y[:, 1:5], y[:, 5:]
        
        l1 = torch.nn.BCEWithLogitsLoss(reduction=self.reduction).forward(obj_indicator_pred, y_obj)
        l2 = torch.nn.MSELoss(reduction=self.reduction).forward(bounding_boxes_pred, y_bbox)
        l3 = torch.nn.CrossEntropyLoss(reduction=self.reduction).forward(classification_pred, y_cls)

        # keep in mind that l2 and l3 are included in the loss only for samples with an object of interest
        final_loss = l1 + y_obj.T @ (l2 + l3)

        return final_loss
