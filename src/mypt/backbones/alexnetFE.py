"""
This script is a wrapper around the pretrained architecture AlexNet provided by Pytorch.
The wrapper allows to extract specific blocks from the architecture and freeze them.
The wrapper also allows to build a classifier on top of the extracted features.

The wrapper is inspired is the analogous to the ResnetFE class.
"""


from torch import nn
from typing import OrderedDict, Union, List
from torchvision.models import alexnet, AlexNet_Weights

from mypt.building_blocks.mixins.custom_module_mixins import WrapperLikeModuleMixin  



class AlexNetFE(WrapperLikeModuleMixin):
    """
    The Alexnet is a small (in modern standards) architecture, offering a more fine-grained control over its different components. The original architecture is split into: 
    * conv1: [nn.Convolution layer, Relu, MaxPool]
    * conv2: [nn.Convolution layer, Relu, MaxPool]
    * conv3: [nn.Convolution layer, Relu]
    * conv4: [nn.Convolution layer, Relu]
    * conv5: [nn.Convolution layer, Relu, MaxPool]
    * adapool: Adaptive Pooling layer
    * fc1: [dropout, linear, relu]
    * fc2: [dropout, linear, relu]
    * fc3: [linear]
    """
    
    __block_name2index = {"conv1":0 , 
                    "conv2":1 , 
                    "conv3":2 , 
                    "conv4":3 , 
                    "conv5":4 , 
                    "avgpool":5 ,
                    }
    
    __index2block_name =  {0: "conv1", 
                    1: "conv2", 
                    2: "conv3", 
                    3: "conv4", 
                    4: "conv5", 
                    5: "avgpool",
                    }

    __list_str_arguments = [f'conv{i}' for i in range(1, 6)] + ['avgpool']
    __str_arguments = ['conv_block', 'conv_block_avgpool'] + __list_str_arguments[:-1]

    @classmethod
    def __verify_blocks(cls, blocks: Union[str, int, List[str], List[int]]) -> List[int]:
        # the `blocks` argument can have one of the following typess / values:
        # 1. a string: representing the last layer of the architecture to add: either `all`, `conv_block`, `conv_block_adapool` or 'conv_{i}` where i from 1 to 5
        # 2. an integer: representing the index of the layer to add: from 0 to 5
        # 3. a list of strings: representing the layers to add: must be one of the following'conv_{i}` where i from 1 to 5 or 'avgpool
        # 4. a list of integers: representing the indices of the layers to add: from 0 to 5
        
        # make sure the argument is one of the expected arguments
        if isinstance(blocks, str):
            # if the argument is a string, then it should be one of the arguments written above
            if blocks not in cls.__str_arguments:
                raise ValueError(f"The initialize received an expected argument: {blocks}. string arguments are expected to be on of {cls.__str_arguments}.")    
            if blocks == cls.__str_arguments[0]:
                return list(range(5)) # return all blocks except the average pooling layer
            elif blocks == cls.__str_arguments[1]:
                return list(range(6)) # return all blocks

            # at this points, `blocks` == [convX]
            n = int(blocks[-1])
            return list(range(n)) # return up to the block with index 'n'


        if isinstance(blocks, int):
            if not ( 0 <= blocks <= 5):
                raise ValueError(f"The initializer received an expected argument: {blocks}. integer arguments are expected to belong to the interval [0, 5].")
            
            return list(range(blocks + 1)) # return up to the block with index 'blocks'
        
        if not isinstance(blocks, (List)):
            raise TypeError(f"The blocks argument is expected to be one of the following types: {str}, {int}, {List[str]}, {List[int]}. Found: {type(blocks)}")

        # make sure all elements are of the same type
        if not (all([isinstance(b, type(blocks[0])) and isinstance(b, (str, int)) for b in blocks])):
            raise TypeError(f"All elements in the list must have the same type. Found elements of different types: {blocks}")

        # if it is a list of strings
        if isinstance(blocks[0], str):
            # each element should be in the cls.__list_str_arguments
            if not (all([b in cls.__list_str_arguments for b in blocks])):
                raise ValueError(f"elements of a string list argument are expected to belong to {cls.__list_str_arguments}. Found: {blocks}")

            # sort the list by their appearance in the '__list_str_arguments'
            sorted_blocks = sorted(blocks, key=lambda x: cls.__list_str_arguments.index(x), reverse=False)

            # map each string element to the index of the block
            return [cls.__block_name2index[b] for b in sorted_blocks]

        
        if isinstance(blocks[0], int):
            if not (all([ 0 <= b <= 5 for b in blocks])):
                raise ValueError(f"The indices are expected to belong to the interval [0, 5]. Found: {blocks}")
            
            return sorted(blocks)

    def __set_convolutional_block(self, convolutional_block: nn.Sequential):
        """
        This function splits the convolutional block introduced by AlexNet into 5 convolutional sub-blocks that can later 
        be addressed and manipulated independently.

        The assumption is that a convolutional block starts with a convolutional layer and ends right before the next convolutional layer 
        (or the end of the convolutional block)
        
        Args:
            convolutional_block (nn.Sequential): the AlexNet convolutional block.
        """
        
        block_index = 0
        current_block = []

        for name, module in convolutional_block.named_children():
            # when we find a convolutional layer, 
            if isinstance(module, nn.Conv2d):
                # if the 'current_block' is empty
                if len(current_block) == 0: 
                    current_block.append((name, module))
                else: 
                    # save the current block and create a new one
                    self._block_indices[block_index] = nn.Sequential(OrderedDict(current_block))
                    current_block = [(name, module)]
                    # increment the block index
                    block_index += 1
            else: 
                current_block.append((name, module))

        # after iterating through the convolutional block, make sure to add the last blok
        self._block_indices[block_index] = nn.Sequential(OrderedDict(current_block))

        # some test to make sure the distribution is as expected
        assert len(self._block_indices) == 5, f"make sure the total number of block is '5'. Found: {len(self._block_indices)}"
        __lengths = [3, 3, 2, 2, 3]

        for l, (index_block, block) in zip(__lengths, self._block_indices.items()):
            assert isinstance(block, nn.Sequential), f"Make sure the block with index {index_block} is a nn.Sequential. Found: {type(block)}"
            assert len(block) == l, f"Make sure the block with index {index_block}. Found: {len(block)}"

    def __set_blocks(self):
        """
        This function maps each basic block to an index so that be accessed easily later on        
        """
        convolutional_block, avgpool, fc_block = list(self.__net.children())
        self.__set_convolutional_block(convolutional_block=convolutional_block)
        self._block_indices[5] = avgpool


    def _build_model(self) -> nn.Sequential:
        # the self._model_blocks is a list of integers that represent the indices of the blocks to be used
        return nn.Sequential(OrderedDict([(self.__index2block_name[i], self._block_indices[i]) for i in self._model_blocks]))
        

    def __verify_frozen_blocks(self,
                            frozen_blocks: Union[bool, str, List[str], int, List[int]]):
        """This function checks the arguments used to determine which blocks to freeze. They are subject to the following considerations: 
        1. if 'frozen_blocks' is a bool variable, then freeze all blocks or leave them as they are

        2. if 'model_blocks' is a 'str', then it should represent a specific layer, it cannot be one of the 'cls.__str_arguments'
            additionally model_blocks must be of 'str' or [str]
        
        3. if 'model_blocks' is 'int', then it has to a value present in 'model_blocks' (or equal) 

        4. 
        Args:
            model_blocks (Union[str, List[str], int, List[int]]): The blocks extracted from the architecture
            frozen_blocks (Union[bool, str, List[str], int, List[int]]): The blocks to be frozen
        """

        if isinstance(frozen_blocks, bool):
            return list(range(6)) if frozen_blocks else []      

        # convert the frozen_blocks to a list of integers
        fb_indices = self.__verify_blocks(frozen_blocks)

        # make sure that frozen_blocks is a subset of model_blocks
        if not set(fb_indices).issubset(self._model_blocks):
            raise ValueError(f"the frozen_blocks are expected to be part of the model blocks. Frozen_blocks: {frozen_blocks}, model_blocks: {self._model_blocks}")

        return fb_indices


    def _freeze_model(self) -> None:
        # self._frozen_blocks is a list of integers !!! (guarnateed by calling the __verify_frozen_blocks function)
        if len(self._frozen_blocks) == 0:
            return 

        for block_index in self._frozen_blocks:
            for p in self._block_indices[block_index].parameters():
                p.requires_grad = False
        
    def __init__(self, 
                model_blocks: Union[str, List[str], int, List[int]],
                frozen_model_blocks: Union[str, List[str], int, List[int]],
                ) -> None:
        
        super().__init__("_model")
        
        # make sure to initialize the Alexnet as a field
        self.__net = alexnet(weights=AlexNet_Weights.DEFAULT)
        # save the default transform that comes with the AlexNet architecture
        self.transform = AlexNet_Weights.DEFAULT.transforms()
        
        # make sure the blocks are passed correctly
        self._model_blocks = self.__verify_blocks(blocks=model_blocks)    

        # make sure the frozen blocks are passed correctly
        self._frozen_blocks = self.__verify_frozen_blocks(frozen_blocks=frozen_model_blocks)
        
        self._block_indices = {}
        self.__set_blocks()

        # freeze the needed layers
        self._freeze_model()

        # build the model
        self._model = self._build_model()

        # delete the __net field and block_indices fields to reduce the size of the class
        del(self.__net)
        del(self._block_indices)
   
