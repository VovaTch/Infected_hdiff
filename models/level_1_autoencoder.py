from typing import Any, List

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBlock1D(nn.Module):
    """
    Double conv block, I leave the change in dimensions to the encoder and decoder classes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2 - 1),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2 - 1),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
        )
            
    def forward(self, x):
        return self.architecture(x)
        
        


class Lvl1Encoder(nn.Module):
    """
    Encoder class for the level 1 auto-encoder, this is constructed in a VAE manner.
    """
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int]):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        self.last_dim = channel_list[-1]
        channel_list[-1] *= 2
        
        # Create the module lists for the architecture
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.MaxPool1d(dim_change_param) for dim_change_param in dim_change_list]
        )
        
    def forward(self, x):
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            x = conv(x)
            x = dim_change(x)
            
        mu = x[:, :self.last_dim, :] # BS x C x W
        sigma = x[:, self.last_dim:, :]
        
        z = torch.randn_like(mu) * sigma + mu
        self.kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1).sum()
        
        return z
    

class Lvl1Decoder(nn.Module):
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int]):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        
        # Create the module lists for the architecture
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.ConvTranspose1d(channel_list[idx + 1], channel_list[idx + 1], 
                                kernel_size=dim_change_list[idx], stride=dim_change_list[idx])
             for idx in range(len(dim_change_list))]
        )
        
    def forward(self, z):
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            z = conv(z)
            z = dim_change(z)
            
        return z


class Lvl1AutoEncoder(pl.LightningModule):
    """
    (VQ later?) VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 slice_length: float,
                 hidden_size: int,
                 latent_depth: int,
                 learning_rate: float,
                 weight_decay: float,
                 **kwargs):
        
        super().__init__()
        
        # Parse arguments
        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.samples_per_slice = int(sample_rate * slice_length)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size, hidden_size * 2, hidden_size * 4, hidden_size * 8, hidden_size * 16, latent_depth]
        encoder_dim_changes = [5, 5, 5, 5, 5]
        decoder_channel_list = encoder_channel_list.reverse()
        decoder_dim_change = [5, 5, 5, 5, 5]
        
        
    def forward(self, x):
        pass
    
    
if __name__ == "__main__":
    
    vae = Lvl1AutoEncoder(44100, 5.0, 64, 16, 0.001, 0.01)
    