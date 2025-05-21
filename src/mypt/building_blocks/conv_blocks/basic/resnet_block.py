import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable


class BasicBlock(nn.Module):
    """Basic residual block for WRN as described in https://arxiv.org/abs/1605.07146
    
    The block uses the pre-activation design (BN+ReLU before convolution)
    and follows the WRN paper's recommendations.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "relu"
    ):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        
        # Set activation function
        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation.lower() == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        
        # First convolution block
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        # Second convolution block
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block (BN -> ReLU -> Conv)
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Dropout if specified
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        # Second block (BN -> ReLU -> Conv)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        # Add shortcut connection
        out += self.shortcut(x)
        
        return out


class ConditionedWideResNetBlock(nn.Module):
    """Conditioned Wide ResNet block that accepts external conditioning information.
    
    This extends the standard WRN block with conditioning capabilities using FiLM
    (Feature-wise Linear Modulation) for adaptive batch normalization.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_channels: int,
        stride: int = 1, 
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "relu",
        conditioning_method: str = "film"  # film, concat, or add
    ):
        super().__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.conditioning_method = conditioning_method
        
        # Set activation function
        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation.lower() == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation
        
        # First convolution block
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        # Second convolution block
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                )
            )
        
        # Conditioning layers
        if conditioning_method == "film":
            # FiLM conditioning generating scale and shift parameters
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_channels * 2)  # 2 for scale and shift
            )
        elif conditioning_method == "add":
            # Simple additive conditioning
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_channels)
            )
        elif conditioning_method == "concat":
            # For concatenation, we'll adjust the second conv to handle more channels
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_channels, out_channels),
                nn.ReLU(inplace=True)
            )
            # Adjust second conv for concatenated inputs
            self.conv2 = nn.Conv2d(
                out_channels * 2,  # Double channels due to concatenation
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        else:
            raise ValueError(f"Unsupported conditioning method: {conditioning_method}")
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First block (BN -> ReLU -> Conv)
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Dropout if specified
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        # Apply conditioning based on method
        if self.conditioning_method == "film":
            # FiLM conditioning (scale and shift)
            cond = self.cond_encoder(condition)
            gamma, beta = torch.chunk(cond, 2, dim=1)
            
            # Reshape for broadcasting
            gamma = gamma.view(gamma.size(0), -1, 1, 1)
            beta = beta.view(beta.size(0), -1, 1, 1)
            
            # Apply after first BN in second block
            out = self.bn2(out)
            out = gamma * out + beta
            out = self.activation(out)
            out = self.conv2(out)
            
        elif self.conditioning_method == "add":
            # Additive conditioning
            cond = self.cond_encoder(condition)
            cond = cond.view(cond.size(0), -1, 1, 1)
            
            out = self.bn2(out)
            out = self.activation(out)
            out = out + cond  # Add conditioning
            out = self.conv2(out)
            
        elif self.conditioning_method == "concat":
            # Concatenation conditioning
            cond = self.cond_encoder(condition)
            cond = cond.view(cond.size(0), -1, 1, 1)
            cond = cond.expand(-1, -1, out.size(2), out.size(3))
            
            out = self.bn2(out)
            out = self.activation(out)
            out = torch.cat([out, cond], dim=1)  # Concatenate along channel dimension
            out = self.conv2(out)
        
        # Add shortcut connection
        out += self.shortcut(identity)
        
        return out


class WideResNet(nn.Module):
    """Implementation of Wide Residual Network as described in https://arxiv.org/abs/1605.07146"""
    def __init__(
        self, 
        depth: int, 
        width_factor: int, 
        num_classes: int, 
        input_channels: int = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        
        # Compute channel widths
        k = width_factor
        nStages = [16, 16*k, 32*k, 64*k]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # Residual blocks
        self.layer1 = self._make_layer(BasicBlock, nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)
        
        # Final BN and classifier
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.bn(out)
        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out