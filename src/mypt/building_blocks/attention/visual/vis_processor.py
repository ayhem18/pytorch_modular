"""
This module prepare a 4 dimensional tensor to be passed into the attention block. The visualAttentionInputProcessor is an integral part of the VisualAttentionBlock.
The processor consists of 

1. a patcher that converts the (bs, c, h, w) tensor into a (bs, n_patches, d_model) tensor 
2. a positional encoder that adds positional information to the patches 
3. 

"""
from torch import nn




class VisAttProcessor(nn.Module):
    def __init__(self, 
                ):
        super().__init__()
        
        
        
class VisAttCondProcessor(nn.Module):
    def __init__(self, 
                ):
        super().__init__()
        
        
        