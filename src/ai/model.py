"""
AI Image Enhancement Neural Network.
A deep learning model for image super-resolution and enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChannelAttention(nn.Module):
    """Channel attention module for focused enhancement."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention for region-aware enhancement."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(out))
        return x * attention


class ImageEnhancerAI(nn.Module):
    """
    Deep neural network for image enhancement.
    Performs super-resolution + quality enhancement in one pass.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_resblocks: int = 16,
        upscale_factor: int = 2,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        self.conv_input = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        self.resblocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_resblocks)
        ])
        
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features)
        
        self.channel_attention = ChannelAttention(num_features)
        self.spatial_attention = SpatialAttention()
        
        if upscale_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )
        else:
            self.upsample = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x):
        x_input = self.conv_input(x)
        residual = self.resblocks(x_input)
        residual = self.bn_mid(self.conv_mid(residual))
        residual = self.channel_attention(residual)
        residual = self.spatial_attention(residual)
        out = residual + x_input
        out = self.upsample(out)
        return out


class LightweightEnhancer(nn.Module):
    """Lightweight version for mobile/fast inference."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, out_channels, 3, padding=1)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        skip = out
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.res1(out)
        out = self.res2(out)
        out = F.relu(self.up(out))
        out = self.conv4(out + skip)
        return out


def create_model(model_type: str = 'standard', device: Optional[str] = None) -> nn.Module:
    """Factory function for AI model creation."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'standard':
        model = ImageEnhancerAI(num_resblocks=16, upscale_factor=2)
    elif model_type == 'lightweight':
        model = LightweightEnhancer()
    elif model_type == 'high_quality':
        model = ImageEnhancerAI(num_resblocks=32, num_features=96, upscale_factor=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)