from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_codebook import VQCodebook
from .base import BaseNetwork
from loss import TotalLoss

class ConvDownsample(nn.Module):
    '''
    A small module handling downsampling via a convolutional layer instead of e.g. Maxpool.
    '''
    
    def __init__(self, kernel_size: int, downsample_divide: int, in_dim: int):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample_divide = downsample_divide
        self.in_dim = in_dim
        self.padding_needed = (kernel_size - 2) + (downsample_divide - 2)
        self.padding_needed = 0 if self.padding_needed < 0 else self.padding_needed # Safeguard against negative padding
        
        # Define the convolutional layer
        self.conv_down = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, stride=downsample_divide)
        
    def forward(self, x):
        x = F.pad(x, (0, self.padding_needed))
        return self.conv_down(x)


class SinActivation(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)


class ConvBlock1D(nn.Module):
    """
    Double conv block, I leave the change in dimensions to the encoder and decoder classes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation_type: str='gelu'):
        super().__init__()
        assert activation_type in ['gelu', 'sin'], 'unknown activation type'
        if activation_type == 'gelu':
            activation_func = nn.GELU()
        elif activation_type == 'sin':
            activation_func = SinActivation()
        self.architecture = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            activation_func,
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            activation_func,
        )
           
            
    def forward(self, x):
        return self.architecture(x)
        


class Encoder1D(nn.Module):
    """
    Encoder class for the level 1 auto-encoder, this is constructed in a VAE manner.
    """
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        self.last_dim = channel_list[-1]
        
        # Create the module lists for the architecture
        self.init_conv = nn.Conv1d(input_channels, channel_list[0], kernel_size=3, padding=1)
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [ConvDownsample(kernel_size=5, downsample_divide=dim_change_param, in_dim=channel_list[idx + 1]) 
             for idx, dim_change_param in enumerate(dim_change_list)]
        )
        
        
    def forward(self, x):
        
        x = self.init_conv(x)
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            x = conv(x)
            x = dim_change(x)
            x = F.gelu(x)
        
        return x


class Decoder1D(nn.Module):
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1, sin_locations: List[int]=None,
                 bottleneck_kernel_size: int=31):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        assert bottleneck_kernel_size % 2 == 1 or bottleneck_kernel_size == 0, \
            f"The bottleneck kernel size {bottleneck_kernel_size} must be an odd number or zero."
        
        self.bottleneck_kernel_size = bottleneck_kernel_size
        
        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(channel_list[-1], input_channels, kernel_size=3, padding=1)
        if bottleneck_kernel_size != 0:
            self.conv_1d_end = nn.Sequential(
                nn.Conv1d(input_channels, channel_list[1], kernel_size=bottleneck_kernel_size, padding=bottleneck_kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(channel_list[1], input_channels, kernel_size=bottleneck_kernel_size, padding=bottleneck_kernel_size // 2)
            )
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5, activation_type='gelu') 
             for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.ConvTranspose1d(channel_list[idx + 1], channel_list[idx + 1], 
                                kernel_size=dim_change_list[idx] + 12, stride=dim_change_list[idx], padding=6)
             for idx in range(len(dim_change_list))]
        )
        self.sin_locations = sin_locations

        
        
    def forward(self, z):
        
        for idx, (conv, dim_change) in enumerate(zip(self.conv_list, self.dim_change_list)):
            z = conv(z)
            z = dim_change(z)
            
            # Trying to have a sinusoidal activation here for repetitive data
            if self.sin_locations is not None:
                if idx + 1 in self.sin_locations:
                    z += torch.sin(z.clone())
            
        x_out = self.end_conv(z)
            
        return x_out + self.conv_1d_end(x_out) if self.bottleneck_kernel_size != 0 else x_out


class VQ1D(nn.Module):
    
    
    def __init__(self, token_dim, num_tokens: int=8192):
        
        super().__init__()
        self.vq_codebook = VQCodebook(token_dim, num_tokens=num_tokens)
        
        
    def forward(self, z_e: torch.Tensor, extract_losses: bool=False):
        
        z_q, indices = self.vq_codebook.apply_codebook(z_e, code_sg=True)
        output = {'indices': indices, 'v_q': z_q}
        
        if extract_losses:
            emb, _ = self.vq_codebook.apply_codebook(z_e.detach())
            output.update({'emb': emb})
            
        return output
        


class MultiLvlVQVariationalAutoEncoder(BaseNetwork):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 slice_time: float,
                 hidden_size: int,
                 latent_depth: int,
                 loss_obj: TotalLoss=None,
                 vocabulary_size: int=8192,
                 bottleneck_kernel_size: int=31,
                 input_channels: int=1,
                 sin_locations: List[int] = None,
                 channel_dim_change_list: List[int] = [2, 2, 2, 4, 4],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Parse arguments
        self.loss_obj = loss_obj
        self.cfg = kwargs
        self.sample_rate = sample_rate
        self.slice_time = slice_time
        self.samples_per_slice = int(sample_rate * slice_time)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.vocabulary_size = vocabulary_size
        self.channel_dim_change_list = channel_dim_change_list
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size * (2 ** (idx + 1)) for idx in range(len(channel_dim_change_list))] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = list(reversed(encoder_channel_list))
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        
        # Initialize network parts
        self.input_channels = input_channels
        self.encoder = Encoder1D(encoder_channel_list, encoder_dim_changes, input_channels=input_channels)
        self.decoder = Decoder1D(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations, 
                                   bottleneck_kernel_size=bottleneck_kernel_size, input_channels=input_channels)
        self.vq_module = VQ1D(latent_depth, num_tokens=vocabulary_size)
        
        
    def forward(self, x: torch.Tensor):
        
        origin_shape = x.shape
        x_reshaped = x.reshape((x.shape[0], -1, self.input_channels)).permute((0, 2, 1))
        
        z_e = self.encoder(x_reshaped)
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'])
        
        total_output = {**vq_block_output,
                        'output': x_out.permute((0, 2, 1)).reshape(origin_shape)}
        
        loss_target = {'music_slice': x,
                       'z_e': z_e}
        
        if self.loss_obj is not None:
            total_output.update(self.loss_obj(total_output, loss_target))
        
        return total_output
    
    
    def training_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice)
        total_loss = total_output['total_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        
        return total_loss
    
    
    def on_train_epoch_end(self):
        self.vq_module.vq_codebook.random_restart()
        self.vq_module.vq_codebook.reset_usage()
    
    
    def validation_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice)
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                prog_bar = True if key == 'total_loss' else False
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value, prog_bar=prog_bar)