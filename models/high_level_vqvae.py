from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .vq_codebook import VQCodebook


class ConvBlock2D(nn.Module):
    """
    Double convolution block 2D, more standard one, as the data from the lower levels expected to be 2D.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.architecture(x)
    
    
class HighLvlEncoder(nn.Module):
    """
    Encoder of 2D data, variation of standard VQ-VAE for vision.
    """
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        self.last_dim = channel_list[-1]
        
                # Create the module lists for the architecture
        self.init_conv = nn.Conv2d(input_channels, channel_list[0], kernel_size=(3, 3), padding=1)
        self.conv_list = nn.ModuleList(
            [ConvBlock2D(channel_list[idx], channel_list[idx + 1], 3) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.MaxPool1d(dim_change_param) for dim_change_param in dim_change_list]
        )
        
    
    def forward(self, x):
        
        x = self.init_conv(x)
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            x = conv(x)
            x = dim_change(x)
        
        return x
    
    
class HighLvlDecoder(nn.Module):
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1, sin_locations: List[int]=None):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        
        # Create the module lists for the architecture
        self.end_conv = nn.Conv2d(channel_list[-1], input_channels, kernel_size=(3, 3), padding=1)
        self.conv_list = nn.ModuleList(
            [ConvBlock2D(channel_list[idx], channel_list[idx + 1], 5) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.ConvTranspose2d(channel_list[idx + 1], channel_list[idx + 1], 
                                kernel_size=(dim_change_list[idx], dim_change_list[idx]), 
                                stride=dim_change_list[idx])
             for idx in range(len(dim_change_list))]
        )
        self.sin_locations = sin_locations

        
    def forward(self, z: torch.Tensor):
        
        for idx, (conv, dim_change) in enumerate(zip(self.conv_list, self.dim_change_list)):
            z = conv(z)
            z = dim_change(z)
            
            # Trying to have a sinusoidal activation here for repetitive data
            if self.sin_locations is not None:
                if idx + 1 in self.sin_locations:
                    z += torch.sin(z.clone())
            
        x_out = self.end_conv(z)
            
        return torch.tanh(x_out)