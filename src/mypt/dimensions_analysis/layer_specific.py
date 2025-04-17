"""
This script contains functionalities to compute the dimensions of the output of several Pytorch components
"""
import numpy as np
from torch import nn
from typing import Union, Tuple, Any, List
from _collections_abc import Sequence

# user-defined types
three_int_tuple = Tuple[int, int, int]
four_int_tuple = Tuple[int, int, int, int]


# Convolutional layers (only 2d at this point)

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


# Pooling layers (only 2d at this point)

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


# Flatten layer
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



# Linear layer
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


# Normalization layers
def batchnorm1d_output(input_shape: Union[Tuple[int, int], Tuple[int, int, int]], 
                      bn_layer: nn.BatchNorm1d) -> Union[Tuple[int, int], Tuple[int, int, int]]:
    """
    This function computes the output dimensions of a BatchNorm1d layer.
    BatchNorm1d preserves the input dimensions.
    """
    if not isinstance(bn_layer, nn.BatchNorm1d):
        raise TypeError(f"'The bn_layer' is expected to be of type: nn.BatchNorm1d\n"
                        f"Found: {type(bn_layer)}")
    
    # BatchNorm1d expects input of shape (batch_size, features) or (batch_size, features, seq_len)
    if len(input_shape) not in [2, 3]:
        raise ValueError(f"BatchNorm1d expects input of shape (batch_size, features) or "
                         f"(batch_size, features, seq_len). Found shape: {input_shape}")
    
    # For BatchNorm1d, the second dimension (features) should match num_features
    if input_shape[1] != bn_layer.num_features:
        raise ValueError(f"BatchNorm1d expects {bn_layer.num_features} features, but got {input_shape[1]}")
    
    # BatchNorm1d preserves the input dimensions
    return input_shape


def batchnorm2d_output(input_shape: Union[three_int_tuple, four_int_tuple],
                      bn_layer: nn.BatchNorm2d) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the output dimensions of a BatchNorm2d layer.
    BatchNorm2d preserves the input dimensions.
    """
    if not isinstance(bn_layer, nn.BatchNorm2d):
        raise TypeError(f"'The bn_layer' is expected to be of type: nn.BatchNorm2d\n"
                        f"Found: {type(bn_layer)}")
    
    # BatchNorm2d expects input of shape (batch_size, channels, height, width)
    if len(input_shape) not in [3, 4]:
        raise ValueError(f"BatchNorm2d expects input of shape (batch_size, channels, height, width) or "
                         f"(channels, height, width). Found shape: {input_shape}")
    
    # Check if the number of channels matches
    channels_index = 0 if len(input_shape) == 3 else 1
    if input_shape[channels_index] != bn_layer.num_features:
        raise ValueError(f"BatchNorm2d expects {bn_layer.num_features} channels, "
                         f"but got {input_shape[channels_index]}")
    
    # BatchNorm2d preserves the input dimensions
    return input_shape


def layernorm_output(input_shape: Tuple[int, ...], ln_layer: nn.LayerNorm) -> Tuple[int, ...]:
    """
    This function computes the output dimensions of a LayerNorm layer.
    LayerNorm preserves the input dimensions.
    """
    if not isinstance(ln_layer, nn.LayerNorm):
        raise TypeError(f"'The ln_layer' is expected to be of type: nn.LayerNorm\n"
                        f"Found: {type(ln_layer)}")
    
    # Check if the normalized shape matches with the input shape
    normalized_shape = ln_layer.normalized_shape
    
    # Convert normalized_shape to tuple if it's a single int
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    # Check if the last dimensions of input_shape match normalized_shape
    if input_shape[-len(normalized_shape):] != normalized_shape:
        raise ValueError(f"LayerNorm expects last dimensions to be {normalized_shape}, "
                         f"but got {input_shape[-len(normalized_shape):]}")
    
    # LayerNorm preserves the input dimensions
    return input_shape


def groupnorm_output(input_shape: Union[three_int_tuple, four_int_tuple],
                    gn_layer: nn.GroupNorm) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the output dimensions of a GroupNorm layer.
    GroupNorm preserves the input dimensions.
    """
    if not isinstance(gn_layer, nn.GroupNorm):
        raise TypeError(f"'The gn_layer' is expected to be of type: nn.GroupNorm\n"
                        f"Found: {type(gn_layer)}")
    
    # GroupNorm expects input of shape (batch_size, channels, *) or (channels, *)
    if len(input_shape) < 2:
        raise ValueError(f"GroupNorm expects input of shape (batch_size, channels, ...) or "
                         f"(channels, ...). Found shape: {input_shape}")
    
    # Check if the number of channels is divisible by the number of groups
    channels_index = 0 if len(input_shape) == 3 else 1
    if input_shape[channels_index] != gn_layer.num_channels:
        raise ValueError(f"GroupNorm expects {gn_layer.num_channels} channels, "
                         f"but got {input_shape[channels_index]}")
    
    if input_shape[channels_index] % gn_layer.num_groups != 0:
        raise ValueError(f"Number of channels ({input_shape[channels_index]}) should be divisible "
                         f"by number of groups ({gn_layer.num_groups})")
    
    # GroupNorm preserves the input dimensions
    return input_shape


def instancenorm2d_output(input_shape: Union[three_int_tuple, four_int_tuple],
                         in_layer: nn.InstanceNorm2d) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the output dimensions of an InstanceNorm2d layer.
    InstanceNorm2d preserves the input dimensions.
    """
    if not isinstance(in_layer, nn.InstanceNorm2d):
        raise TypeError(f"'The in_layer' is expected to be of type: nn.InstanceNorm2d\n"
                        f"Found: {type(in_layer)}")
    
    # InstanceNorm2d expects input of shape (batch_size, channels, height, width) or (channels, height, width)
    if len(input_shape) not in [3, 4]:
        raise ValueError(f"InstanceNorm2d expects input of shape (batch_size, channels, height, width) or "
                         f"(channels, height, width). Found shape: {input_shape}")
    
    # Check if the number of channels matches
    channels_index = 0 if len(input_shape) == 3 else 1
    if in_layer.num_features != 0 and input_shape[channels_index] != in_layer.num_features:
        raise ValueError(f"InstanceNorm2d expects {in_layer.num_features} channels, "
                         f"but got {input_shape[channels_index]}")
    
    # InstanceNorm2d preserves the input dimensions
    return input_shape


# Dropout layers
def dropout_output(input_shape: Tuple[int, ...], dropout_layer: nn.Dropout) -> Tuple[int, ...]:
    """
    This function computes the output dimensions of a Dropout layer.
    Dropout preserves the input dimensions.
    """
    if not isinstance(dropout_layer, nn.Dropout):
        raise TypeError(f"'The dropout_layer' is expected to be of type: nn.Dropout\n"
                        f"Found: {type(dropout_layer)}")
    
    # Dropout preserves the input dimensions
    return input_shape


def dropout2d_output(input_shape: Union[three_int_tuple, four_int_tuple],
                    dropout_layer: nn.Dropout2d) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the output dimensions of a Dropout2d layer.
    Dropout2d preserves the input dimensions.
    """
    if not isinstance(dropout_layer, nn.Dropout2d):
        raise TypeError(f"'The dropout_layer' is expected to be of type: nn.Dropout2d\n"
                        f"Found: {type(dropout_layer)}")
    
    # Dropout2d expects input of shape (batch_size, channels, height, width) or (channels, height, width)
    if len(input_shape) not in [3, 4]:
        raise ValueError(f"Dropout2d expects input of shape (batch_size, channels, height, width) or "
                         f"(channels, height, width). Found shape: {input_shape}")
    
    # Dropout2d preserves the input dimensions
    return input_shape


# Activation functions
def activation_output(input_shape: Tuple[int, ...], activation_layer: Any) -> Tuple[int, ...]:
    """
    This function computes the output dimensions of activation layers like ReLU, LeakyReLU, Sigmoid, etc.
    Activation functions preserve the input dimensions.
    """
    # Common activation types
    activation_types = (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.SELU, 
                        nn.CELU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Hardtanh, 
                        nn.Hardsigmoid, nn.Softplus, nn.Softsign, nn.SiLU)
    
    if not isinstance(activation_layer, activation_types):
        raise TypeError(f"'The activation_layer' is expected to be an activation function\n"
                        f"Found: {type(activation_layer)}")
    
    # All activation functions preserve the input dimensions
    return input_shape


# Upsample and ConvTranspose2d
def upsample_output(input_shape: Union[three_int_tuple, four_int_tuple],
                   upsample_layer: nn.Upsample) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the output dimensions of an Upsample layer.
    """
    if not isinstance(upsample_layer, nn.Upsample):
        raise TypeError(f"'The upsample_layer' is expected to be of type: nn.Upsample\n"
                        f"Found: {type(upsample_layer)}")
    
    # Upsample expects input of shape (batch_size, channels, height, width) or (channels, height, width)
    if len(input_shape) not in [3, 4]:
        raise ValueError(f"Upsample expects input of shape (batch_size, channels, height, width) or "
                         f"(channels, height, width). Found shape: {input_shape}")
    
    batch_size, channels, height, width = None, None, None, None
    
    if len(input_shape) == 4:
        if upsample_layer.mode == 'linear':
            raise ValueError("The upsample layer expects 3D input when the mode is set to 'linear'")
        batch_size, channels, height, width = input_shape
    else:
        channels, height, width = input_shape
    
    # Calculate new height and width based on scale_factor or size
    if upsample_layer.scale_factor is not None:
        scale_factor = upsample_layer.scale_factor
        if isinstance(scale_factor, tuple):
            new_height = int(height * scale_factor[0])
            new_width = int(width * scale_factor[1])
        else:
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
    else:  # size is specified
        size = upsample_layer.size
        if isinstance(size, tuple):
            if len(size) == 1:
                new_height = size[0]
                new_width = size[0]
            else:
                new_height, new_width = size
        else:
            new_height = size
            new_width = size
    
    result_shape = (batch_size,) if batch_size is not None else tuple()
    result_shape += (channels, new_height, new_width)
    
    return result_shape


def __convtranspose2d_output2D(height: int, width: int, conv_layer: nn.ConvTranspose2d) -> tuple[int, int]:
    """
    This function computes the output dimensions of a 2d ConvTranspose layer
    Only height and width are considered
    """
    # Extract the parameters
    s1, s2 = conv_layer.stride
    k1, k2 = conv_layer.kernel_size
    p1, p2 = conv_layer.padding
    op1, op2 = conv_layer.output_padding
    d1, d2 = conv_layer.dilation
    
    # Apply the formula from PyTorch documentation
    new_h = (height - 1) * s1 - 2 * p1 + d1 * (k1 - 1) + op1 + 1
    new_w = (width - 1) * s2 - 2 * p2 + d2 * (k2 - 1) + op2 + 1
    
    return new_h, new_w


def convtranspose2d_output(input_shape: Union[three_int_tuple, four_int_tuple],
                          conv_layer: nn.ConvTranspose2d) -> Union[three_int_tuple, four_int_tuple]:
    """
    This function computes the shape of the output of a 2d ConvTranspose layer
    """
    if not isinstance(input_shape, Sequence):
        raise TypeError("The 'input_shape' argument is expected to be an iterable preferably a tuple \n"
                        f"Found the type: {type(input_shape)}")
    
    if len(input_shape) not in [3, 4]:
        raise ValueError("The 'input_shape' argument must be either 3 or 4 dimensional \n"
                         f"Found: an input of size: {len(input_shape)}")
    
    if not isinstance(conv_layer, nn.ConvTranspose2d):
        raise TypeError("The 'conv_layer' is expected to a be a 2D convolutional transpose layer\n"
                        f"Found the type: {type(conv_layer)}")
    
    batch, channels, height, width = None, None, None, None
    
    if len(input_shape) == 4:
        batch, channels, height, width = input_shape
    else:
        channels, height, width = input_shape
    
    # Check if the number of input channels matches
    if channels != conv_layer.in_channels:
        raise ValueError(f"ConvTranspose2d expects {conv_layer.in_channels} input channels, but got {channels}")
    
    # Extract the new height and width
    new_h, new_w = __convtranspose2d_output2D(height, width, conv_layer)
    
    # Extract the number of output channels from the layer
    new_channels = conv_layer.out_channels
    
    result_shape = (batch,) if batch is not None else tuple()
    result_shape += (new_channels, new_h, new_w)
    
    return result_shape


# Embedding layer
def embedding_output(input_shape: Tuple[int, ...], embedding_layer: nn.Embedding) -> Tuple[int, ...]:
    """
    This function computes the output dimensions of an Embedding layer.
    """
    if not isinstance(embedding_layer, nn.Embedding):
        raise TypeError(f"'The embedding_layer' is expected to be of type: nn.Embedding\n"
                        f"Found: {type(embedding_layer)}")
    
    # Embedding expects input of any shape with integer values
    # The output shape will be the same as input, but with an extra dimension of size embedding_dim
    return input_shape + (embedding_layer.embedding_dim,)