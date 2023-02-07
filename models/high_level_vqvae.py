from typing import List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .vq_codebook import VQCodebook
from .level_1_vqvae import Lvl1VQ as HighLvlVQ
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
            
        return x_out
    
    
class HighLvlVQVariationalAutoEncoder(pl.LightningModule):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 slice_length: float,
                 hidden_size: int,
                 latent_depth: int,
                 learning_rate: float,
                 weight_decay: float,
                 batch_size: int,
                 epochs: int,
                 beta_factor: float=0.5,
                 vocabulary_size: int=8192,
                 channel_dim_change_list: List[int] = [2, 2, 2, 2, 2, 2],
                 dataset_name: str='lvl1_out',
                 dataset_path: str='data/vqvae_lvl1_out',
                 optimizer_name: str='one_cycle_lr',
                 eval_split_factor: float=0.01,
                 **kwargs):
        
        super().__init__()
        
        # Parse arguments
        self.cfg = kwargs
        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.samples_per_slice = int(sample_rate * slice_length)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta_factor = beta_factor
        self.dataset_path = dataset_path
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
        self.decoder = HighLvlDecoder(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations)
        self.vq_module = HighLvlVQ(latent_depth, num_tokens=vocabulary_size)
    
    
        # Datasets
        assert dataset_name in DATASETS, f'Dataset {dataset_name} is not in the datasets options.'
        assert 0 <= eval_split_factor <= 1, f'The split factor must be between 0 and 1, current value is {eval_split_factor}'
        self.dataset = None
        self.eval_split_factor = eval_split_factor
        self.dataset_name = dataset_name
        
        # Optimizers
        assert optimizer_name in ['none', 'one_cycle_lr', 'reduce_on_platou'] # TODO fix typo, program the schedulers in