import random
import torch.nn as nn

from random import randint as ri


class RandomLayerGenerator:
    def __init__(self):
        pass


    def generate_random_conv_layer(self) -> nn.Conv2d:
        """
        This function generates a random Conv2d layer with random parameters
        """
        # Create random parameters for Conv2d
        in_channels = ri(1, 64)
        out_channels = ri(1, 64)
        kernel_size = ri(1, 7) 
        stride = ri(1, 3)
        padding = ri(0, 3)
        dilation = ri(1, 2)
        
        # Create a Conv2d layer with these parameters
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        return conv_layer        

    def generate_random_avg_pool_layer(self) -> nn.AvgPool2d:
        """
        This function generates a random AvgPool2d layer with random parameters
        """
        kernel_size = ri(3, 11)
        stride = ri(1, 3)
        padding = ri(0, 2)
        padding = min(padding, kernel_size // 2)

        # Create an AvgPool2d layer with these parameters
        pool_layer = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        return pool_layer
        
    def generate_random_max_pool_layer(self) -> nn.MaxPool2d:
        """
        This function generates a random MaxPool2d layer with random parameters
        """
        kernel_size = ri(3, 11)
        stride = ri(1, 3)
        padding = random.choice([0, 1, 2])
        padding = min(padding, kernel_size // 2)

        dilation = ri(1, 2)
        
        # Create a MaxPool2d layer with these parameters
        pool_layer = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        return pool_layer
        
    def generate_random_adaptive_pool_layer(self) -> nn.AdaptiveAvgPool2d:
        """
        This function generates a random AdaptiveAvgPool2d layer with random parameters
        """
        output_height = ri(1, 10)
        output_width = ri(1, 10)
        
        # Create an AdaptiveAvgPool2d layer with these parameters
        pool_layer = nn.AdaptiveAvgPool2d((output_height, output_width))
        
        return pool_layer
        
    def generate_random_linear_layer(self) -> nn.Linear:
        """
        This function generates a random Linear layer with random parameters
        """
        in_features = ri(1, 100)
        out_features = ri(1, 100)
        
        # Create a Linear layer with these parameters
        linear_layer = nn.Linear(in_features, out_features)
        
        return linear_layer
        
    def generate_random_flatten_layer(self) -> nn.Flatten:
        """
        This function generates a random Flatten layer with random parameters
        """
        start_dim = random.choice([1, 2])
        end_dim = -1
        
        # Create a Flatten layer with these parameters
        flatten_layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        
        return flatten_layer

    
    def generate_random_batchnorm1d_layer(self) -> nn.BatchNorm1d:
        """Generate a random BatchNorm1d layer with random parameters"""
        num_features = ri(1, 64)
        eps = random.choice([1e-5, 1e-4, 1e-3])
        momentum = random.choice([0.1, 0.01, 0.001])
        affine = random.choice([True, False])
        track_running_stats = random.choice([True, False])
        
        bn_layer = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        
        return bn_layer
    
    def generate_random_batchnorm2d_layer(self) -> nn.BatchNorm2d:
        """Generate a random BatchNorm2d layer with random parameters"""
        num_features = ri(1, 64)
        eps = random.choice([1e-5, 1e-4, 1e-3])
        momentum = random.choice([0.1, 0.01, 0.001])
        affine = random.choice([True, False])
        track_running_stats = random.choice([True, False])
        
        bn_layer = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        
        return bn_layer
    
    def generate_random_layernorm_layer(self, input_dim1=None, input_dim2=None) -> nn.LayerNorm:
        """Generate a random LayerNorm layer with random parameters"""
        if input_dim1 is None:
            normalized_shape = (ri(1, 32), ri(1, 32))
        elif input_dim2 is None:
            normalized_shape = (input_dim1,)
        else:
            normalized_shape = (input_dim1, input_dim2)
            
        eps = random.choice([1e-5, 1e-4, 1e-3])
        elementwise_affine = random.choice([True, False])
        
        ln_layer = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
        
        return ln_layer
    
    def generate_random_groupnorm_layer(self) -> nn.GroupNorm:
        """Generate a random GroupNorm layer with random parameters"""
        num_channels = ri(4, 64)
        # Make sure num_groups divides num_channels
        possible_groups = [g for g in [1, 2, 4, 8, 16, 32] if num_channels % g == 0]
        num_groups = random.choice(possible_groups)
        
        eps = random.choice([1e-5, 1e-4, 1e-3])
        affine = random.choice([True, False])
        
        gn_layer = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        )
        
        return gn_layer
    
    def generate_random_instancenorm2d_layer(self) -> nn.InstanceNorm2d:
        """Generate a random InstanceNorm2d layer with random parameters"""
        num_features = ri(1, 64)
        eps = random.choice([1e-5, 1e-4, 1e-3])
        momentum = random.choice([0.1, 0.01, 0.001])
        affine = random.choice([True, False])
        track_running_stats = random.choice([True, False])
        
        in_layer = nn.InstanceNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        
        return in_layer
    
    def generate_random_dropout_layer(self) -> nn.Dropout:
        """Generate a random Dropout layer with random parameters"""
        p = random.uniform(0.1, 0.9)
        
        dropout_layer = nn.Dropout(p=p)
        
        return dropout_layer
    
    def generate_random_dropout2d_layer(self) -> nn.Dropout2d:
        """Generate a random Dropout2d layer with random parameters"""
        p = random.uniform(0.1, 0.9)
        
        dropout_layer = nn.Dropout2d(p=p)
        
        return dropout_layer
    
    def generate_random_activation_layer(self):
        """Generate a random activation layer"""
        activation_class = random.choice([nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid, nn.Tanh])
        
        if activation_class == nn.LeakyReLU:
            negative_slope = random.uniform(0.01, 0.3)
            return activation_class(negative_slope=negative_slope)
        else:
            return activation_class()
    
    def generate_random_upsample_layer(self) -> nn.Upsample:
        """Generate a random Upsample layer with random parameters"""
        mode = random.choice(['nearest', 'bilinear', 'bicubic'])
        
        # Choose between scale_factor and size
        if random.choice([True, False]):
            # Use scale_factor
            scale_factor = random.randint(2, 4)
            upsample_layer = nn.Upsample(scale_factor=scale_factor, mode=mode)
        else:
            # Use size
            size = (random.randint(16, 32), random.randint(16, 32))
            upsample_layer = nn.Upsample(size=size, mode=mode)
        
        return upsample_layer
    
    def generate_random_convtranspose2d_layer(self) -> nn.ConvTranspose2d:
        """Generate a random ConvTranspose2d layer with random parameters"""
        in_channels = ri(1, 64)
        out_channels = ri(1, 64)
        kernel_size = ri(1, 7)
        stride = ri(1, 3)
        padding = ri(0, 3)
        output_padding = ri(0, min(stride - 1, 2))  # Must be less than stride
        dilation = ri(1, 2)
        
        conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation
        )
        
        return conv_layer
    
    def generate_random_embedding_layer(self) -> nn.Embedding:
        """Generate a random Embedding layer with random parameters"""
        num_embeddings = ri(10, 1000)
        embedding_dim = ri(8, 64)
        
        embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        
        return embedding_layer
