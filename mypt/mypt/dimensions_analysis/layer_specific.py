"""
This script contains functionalities to compute the dimensions of the output of several Pytorch components
"""
import numpy as np
from torch import nn
from typing import Union, Tuple
from _collections_abc import Sequence

# user-defined types
three_int_tuple = Tuple[int, int, int]
four_int_tuple = Tuple[int, int, int, int]


def __conv2d_output2D(height: int, width: int, conv_layer: nn.Conv2d) -> tuple[int, int]:
    """
    This function computes the output dimensions of a 2d Convolutional layer
    Only height and width are considered as number of channels is not modified by the conv2D layer

    NOTE: this function is not meant to be used directly
    """
    # this code is based on the documentation of conv2D module pytorch:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    # extract the numerical features first
    s1, s2 = conv_layer.stride
    k1, k2 = conv_layer.kernel_size
    d1, d2 = conv_layer.dilation

    # the padding is tricky
    if conv_layer.padding == 'same':
        return height, width

    if conv_layer.padding == 'valid':
        # 'valid' means there is no padding
        p1, p2 = 0, 0

    else:
        p1, p2 = conv_layer.padding

    new_h = int((height + 2 * p1 - d1 * (k1 - 1) - 1) / s1) + 1
    new_w = int((width + 2 * p2 - d2 * (k2 - 1) - 1) / s2) + 1

    return new_h, new_w


def conv2d_output(input_shape: Union[three_int_tuple, four_int_tuple],
                  conv_layer: nn.Conv2d) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the shape of the output of a 2d convolutional layer, given a 3-dimensional input
    """
    if not isinstance(input_shape, Sequence):
        raise TypeError("The 'input_shape' argument is expected to be an iterable preferably a tuple \n"
                        f"Found: {type(input_shape)}")

    if len(input_shape) not in [3, 4]:
        # according  to the documentation of Errors in python, ValueError is the most appropriate in this case
        raise ValueError("The 'input_shape' argument must be either 3 or 4 dimensional \n"
                         f"Found: an input of size: {len(input_shape)}")

    if not isinstance(conv_layer, nn.Conv2d):
        raise TypeError("The 'conv_layer' argument is expected to a be a 2D convolutional layer\n"
                        f"Found the type: {type(conv_layer)}")

    batch, channels, height, width = (None,) * 4
    if len(input_shape) == 4:
        batch, channels, height, width = input_shape
    else:
        channels, height, width = input_shape

    # extracting the new height and width
    new_h, new_w = __conv2d_output2D(height, width, conv_layer)
    # extracting the number of channels from the convolutional layer object
    new_channels = conv_layer.out_channels

    result_shape = (batch,) if batch is not None else tuple()
    result_shape += (new_channels, new_h, new_w)
    return result_shape


def __avg_pool2d_output2D(height: int, width: int, pool_layer: nn.AvgPool2d) -> tuple[int, int]:
    """
    This function computes the output dimensions of a 2d average pooling layer
    Only height and width are considered as the rest of dimensions are not modified by the layer
    This function is based on the documentation of torch.nn.AvgPool2d:
    https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    """

    # extract the kernel information: the typing is important
    kernel = pool_layer.kernel_size if isinstance(pool_layer.kernel_size, Tuple) else \
        (pool_layer.kernel_size, pool_layer.kernel_size)
    k1, k2 = kernel

    # extract the padding information
    # TODO:READ MORE ABOUT THE PADDING DEFAULT VALUES
    # (there is a mention of infinity values which should be carefully considered)
    padding = pool_layer.padding if isinstance(pool_layer.padding, Tuple) else \
        (pool_layer.padding, pool_layer.padding)
    p1, p2 = padding

    # extract stride information
    stride = pool_layer.stride if isinstance(pool_layer.stride, tuple) else (pool_layer.stride, pool_layer.stride)
    s1, s2 = stride

    new_h = int((height + 2 * p1 - k1) // s1) + 1
    new_w = int((width + 2 * p2 - k2) // s2) + 1

    return new_h, new_w


def __max_pool2d_output2D(height: int, width: int, pool_layer: Union[nn.MaxPool2d, nn.AvgPool2d]) -> tuple[int, int]:
    """
    This function computes the output dimensions of a 2d MAx pooling layer
    Only height and width are considered as the rest of dimensions are not modified by the layer
    This function is based on the documentation of torch.nn.MaxPool2d:
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    """
    # extract the kernel information: the typing is important
    kernel = pool_layer.kernel_size if isinstance(pool_layer.kernel_size, Tuple) else \
        (pool_layer.kernel_size, pool_layer.kernel_size)
    k1, k2 = kernel

    # extract the padding information
    # TODO:READ MORE ABOUT THE PADDING DEFAULT VALUES
    # (there is a mention of infinity values which should be carefully considered)
    padding = pool_layer.padding if isinstance(pool_layer.padding, Tuple) else \
        (pool_layer.padding, pool_layer.padding)
    p1, p2 = padding

    # extract the dilation information
    dilation = pool_layer.dilation if isinstance(pool_layer.dilation, Tuple) else \
        (pool_layer.dilation, pool_layer.dilation)
    d1, d2 = dilation

    # extract stride information
    stride = pool_layer.stride if isinstance(pool_layer.stride, tuple) else (pool_layer.stride, pool_layer.stride)
    s1, s2 = stride

    new_h = int((height + 2 * p1 - d1 * (k1 - 1) - 1) / s1) + 1
    new_w = int((width + 2 * p2 - d2 * (k2 - 1) - 1) / s2) + 1
    return new_h, new_w


def pool2d_output(input_shape: Union[four_int_tuple, three_int_tuple],
                  pool_layer: Union[nn.AvgPool2d, nn.MaxPool2d]) -> Union[four_int_tuple, three_int_tuple]:
    if not isinstance(input_shape, Sequence):
        raise TypeError("The 'input_shape' argument is expected to be an iterable preferably a tuple \n"
                        f"Found the type: {type(input_shape)}")

    if len(input_shape) not in [3, 4]:
        # according  to the documentation of Errors in python, ValueError is the most appropriate in this case
        raise ValueError("The 'input_shape' argument must be either 3 or 4 dimensional \n"
                         f"Found: an input of size: {len(input_shape)}")

    if not isinstance(pool_layer, (nn.AvgPool2d, nn.MaxPool2d)):
        raise TypeError("The 'pool_layer' is expected to a be a pooling layer \n"
                        f"Found the type: {type(pool_layer)}")

    batch, channels, height, width = (None,) * 4
    if len(input_shape) == 4:
        batch, channels, height, width = input_shape
    else:
        channels, height, width = input_shape

    # the output will depend on the exact type of the pooling layer
    if isinstance(pool_layer, nn.MaxPool2d):
        # extracting the new height and width
        new_h, new_w = __max_pool2d_output2D(height, width, pool_layer)
    else:
        new_h, new_w = __avg_pool2d_output2D(height, width, pool_layer)

    # the pooling layers do not change the number of channels
    result_shape = (batch,) if batch is not None else tuple()
    result_shape += (channels, new_h, new_w)
    return result_shape


def adaptive_pool2d_output(input_shape: Union[four_int_tuple, three_int_tuple],
                           pool_layer: Union[nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d]) \
        -> Union[four_int_tuple, three_int_tuple]:
    if not isinstance(input_shape, Sequence):
        raise TypeError("The 'input_shape' argument is expected to be an iterable preferably a tuple \n"
                        f"Found the type: {type(input_shape)}")

    if len(input_shape) not in [3, 4]:
        # according  to the documentation of Errors in python, ValueError is the most appropriate in this case
        raise ValueError("The 'input_shape' argument must be either 3 or 4 dimensional \n"
                         f"Found: an input of size: {len(input_shape)}")

    if not isinstance(pool_layer, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        raise TypeError("The 'pool_layer' is expected to a be an ADAPTIVE pooling layer \n"
                        f"Found the type: {type(pool_layer)}")

    batch, channels, height, width = (None,) * 4
    if len(input_shape) == 4:
        batch, channels, _, _ = input_shape
    else:
        channels, _, _ = input_shape

    # the output dimensions are independent of the input's dimensions
    result_shape = (batch,) if batch is not None else tuple()

    new_h, new_w = pool_layer.output_size if isinstance(pool_layer.output_size, tuple) \
        else (pool_layer.output_size, pool_layer.output_size)

    result_shape += (channels, new_h, new_w)
    return result_shape


def flatten_output(input_shape: Tuple, flatten_layer: nn.Flatten) -> int:
    """
    This function computes the output dimensions of a Flatten layer.
    It is based on the official documentation of torch.nn.Flatten():
    https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    """
    # extract the start and end fields
    start, end = flatten_layer.start_dim, flatten_layer.end_dim
    # the end dim can be assigned a negative index:
    end += len(input_shape)

    # the input shape can be broken into 3 parts: the flattened part, the one before it and the one after it
    before_flattened = input_shape[:start]

    flattened = np.prod(input_shape[start: end + 1], dtype=np.intc)
    flattened = flattened.item() if isinstance(flattened, np.ndarray) else flattened
    flattened = (flattened,)

    after_flattened = input_shape[end + 1] if end + 1 < len(input_shape) else tuple()

    return before_flattened + flattened + after_flattened


def linear_output(input_shape: int, linear_layer: nn.Linear) -> int:
    if not isinstance(linear_layer, nn.Linear):
        raise TypeError(f"'The linear_layer' is expected to be of type: nn.Linear\n"
                        f"Found: {type(linear_layer)}")

    if isinstance(input_shape, Tuple):
        if len(input_shape) != 2: 
            raise ValueError(f"the input is expected to be at most 2 dimensional. Found: {len(input_shape)}")
        
        if linear_layer.in_features != input_shape[1]:
            raise ValueError(f"The number of input units expected is: {linear_layer.in_features}.\n"
                         f"Found: {input_shape}")
    
        # the output is independent of the input shape
        return input_shape[0], linear_layer.out_features

    if linear_layer.in_features != input_shape:
        raise ValueError(f"The number of input units expected is: {linear_layer.in_features}.\n"
                        f"Found: {input_shape}")

    return linear_layer.out_features