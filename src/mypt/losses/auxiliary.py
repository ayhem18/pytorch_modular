"""
This script contains the implementation of different losses.
"""

import torch

class MaskSoftmax(torch.nn.Module):
    """
    This class implement the softmax operation given a mask without resorting to passing (-inf) to the built-in pytorch implementation.

    NOTE: there is most likely an implementation that serves the same purpose. However, this library is all about learning and experimenting...
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        # convert mask to boolean dtype if needed
        mask = mask.bool()
        # the function assumes the mask is of shape broadcastable to that of the input tensor

        # the original softmax function is implemented as torch.max(x, dim=dim), then subtracting the max of the input tensor 

        # 1. compute the number of mask elements in the "dim" dimension
        num_masked_elements = (~mask).float().sum(dim=dim, keepdim=True) 
        mask_float = mask.float()
        
        # 2. compute the max of the input tensor in the "dim" dimension (on the non-masked elements...)
        #  max_x computes the max of the non-masked elements in the "dim" dimension
        # the masked elements are set to -inf so they don't contribute to the max

        max_x = torch.max(x.masked_fill(~mask, float('-inf')), dim=dim, keepdim=True).values
        # max_x must not contain "inf" values (which can happen if all elements are masked)
        max_x = max_x.masked_fill(torch.isinf(max_x), 0)

        # 3. compute the numerator (x - max_x) * mask
        numerator = torch.exp((x - max_x) * mask_float)
        # 4. the denominator on the other hand should consider the masked elements as zeros 
        # and that's where the 'num_masked_elements' comes into play: the sum of the mask elements is equal to their number... since e^0 = 1 

        # the denominator is the sum of masked and non-masked elements: if all elemnents in a dimension are masked, the final result will be zero
        # which should be clamped to 1
        # if there is at least one non-masked element, the denominator will be the sum of exp (masked elements) which is larger than 1 (since the max of the masked is set to 0)
        # so in both cases, clamping the dominator by 1 is safe. (test the extreme cases and see how it works anyway)

        denominator = torch.clamp(torch.sum(numerator, dim=dim, keepdim=True) - num_masked_elements, min=1)
        # 5. return the softmax of the input tensor: make sure to mask the values in the numerator as well.
        return (numerator * mask_float) / denominator
