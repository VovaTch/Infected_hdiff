from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from .vq_codebook import VQCodebook
from .level_1_vqvae import Lvl1VQ as HighLvlVQ
from .base import BaseNetwork
import loaders


DATASETS = {'lvl1out': loaders.Lvl1VqvaeOuts,
            'lvl2out': loaders.Lvl2VqvaeOuts}


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
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1, sin_locations: List[int]=None,
                 bottleneck_kernel_size: int=31):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        
        # Create the module lists for the architecture
        self.end_conv = nn.Conv2d(channel_list[-1], input_channels, kernel_size=(3, 3), padding=1)
        self.conv_2d_end = nn.Sequential(
            nn.Conv2d(1, channel_list[1], kernel_size=(bottleneck_kernel_size, bottleneck_kernel_size), 
                      padding=bottleneck_kernel_size // 2),
            nn.GELU(),
            nn.Conv2d(channel_list[1], 1, kernel_size=(bottleneck_kernel_size, bottleneck_kernel_size), 
                      padding=bottleneck_kernel_size // 2)
        )
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
            
        return x_out + self.conv_2d_end(x_out)
    
    
class HighLvlVQVariationalAutoEncoder(BaseNetwork):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 slice_length: float,
                 hidden_size: int,
                 latent_depth: int,
                 beta_factor: float=0.5,
                 vocabulary_size: int=8192,
                 bottleneck_kernel_size: int=5,
                 channel_dim_change_list: List[int] = [2, 2, 2, 2, 2, 2],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Parse arguments
        self.cfg = kwargs
        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.samples_per_slice = int(sample_rate * slice_length)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.beta_factor = beta_factor
        self.vocabulary_size = vocabulary_size
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size * (2 ** idx) for idx in range(len(channel_dim_change_list) + 1)]
        encoder_channel_list[-1] = self.latent_depth
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = list(reversed(encoder_channel_list))
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        sin_locations = None
        
        # Initialize network parts
        self.encoder = HighLvlEncoder(encoder_channel_list, encoder_dim_changes)
        self.decoder = HighLvlDecoder(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations, 
                                      bottleneck_kernel_size=bottleneck_kernel_size)
        self.vq_module = HighLvlVQ(latent_depth, num_tokens=vocabulary_size)
        
    def forward(self, x: torch.Tensor, extract_losses: bool=False):
        
        z_e = self.encoder(x)
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'])
        
        total_output = {**vq_block_output,
                        'output': x_out}
        
        if extract_losses:
            total_output.update({'reconstruction_loss': F.mse_loss(x, x_out)})
        
        return total_output
    
    
    def training_step(self, batch, batch_idx):
        
        music_slice = batch['encoded slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        self.log('Training total loss', total_loss)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        
        music_slice = batch['encoded slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value)
        self.log('Validation total loss', total_loss, prog_bar=True)
    