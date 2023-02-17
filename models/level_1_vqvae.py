from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from .vq_codebook import VQCodebook
from loaders import MP3SliceDataset
from .base import BaseNetwork

DATASETS = {'music_slice_dataset': MP3SliceDataset}

class ConvBlock1D(nn.Module):
    """
    Double conv block, I leave the change in dimensions to the encoder and decoder classes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
           
            
    def forward(self, x):
        return self.architecture(x)
        


class Lvl1Encoder(nn.Module):
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
            [nn.MaxPool1d(dim_change_param) for dim_change_param in dim_change_list]
        )
        
        
    def forward(self, x):
        
        x = self.init_conv(x)
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            x = conv(x)
            x = dim_change(x)
        
        return x
    

class Lvl1Decoder(nn.Module):
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1, sin_locations: List[int]=None):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        
        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(channel_list[-1], input_channels, kernel_size=3, padding=1)
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5) for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.ConvTranspose1d(channel_list[idx + 1], channel_list[idx + 1], 
                                kernel_size=dim_change_list[idx], stride=dim_change_list[idx])
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
            
        return x_out


class Lvl1VQ(nn.Module):
    
    
    def __init__(self, token_dim, num_tokens: int=8192):
        
        super().__init__()
        self.vq_codebook = VQCodebook(token_dim, num_tokens=num_tokens)
        
        
    def forward(self, z_e: torch.Tensor, extract_losses: bool=False):
        
        z_q, indices = self.vq_codebook.apply_codebook(z_e, code_sg=True)
        output = {'indices': indices, 'v_q': z_q}
        
        if extract_losses:
            emb, _ = self.vq_codebook.apply_codebook(z_e.detach())
            output.update({'v_q_detached': emb})
            losses = {'alignment_loss': torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1)),
                      'commitment_loss': torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))}
            output.update(losses)
            
        return output
        


class Lvl1VQVariationalAutoEncoder(BaseNetwork):
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
                 mel_factor: float=100.0,
                 vocabulary_size: int=8192,
                 channel_dim_change_list: List[int] = [2, 2, 2, 4, 4],
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
        self.mel_factor = mel_factor
        
        # Initialize mel spectrogram, TODO: Might do multiple ones for multiple losses
        self.mel_spec = None
        if 'mel_spec_config' in kwargs:
            self.mel_spec_config = kwargs['mel_spec_config']
            self.mel_spec = MelSpectrogram(sample_rate=sample_rate, **self.mel_spec_config)
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size * ((idx + 1) ** 2) for idx in range(len(channel_dim_change_list))] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = list(reversed(encoder_channel_list))
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        sin_locations = None
        
        # Initialize network parts
        self.encoder = Lvl1Encoder(encoder_channel_list, encoder_dim_changes)
        self.decoder = Lvl1Decoder(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations)
        self.vq_module = Lvl1VQ(latent_depth, num_tokens=vocabulary_size)
        
        
    def forward(self, x: torch.Tensor, extract_losses: bool=False):
        
        z_e = self.encoder(x)
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'])
        
        total_output = {**vq_block_output,
                        'output': x_out}
        
        if extract_losses:
            total_output.update({'reconstruction_loss': F.mse_loss(x, x_out)})
            total_output.update({'stft_loss': F.mse_loss(self._mel_spec_and_process(x), 
                                                         self._mel_spec_and_process(x_out))})
        
        return total_output
    
    
    def _mel_spec_and_process(self, x: torch.Tensor):
        """
        To prepare the mel spectrogram loss, everything needs to be prepared.

        Args:
            x (torch.Tensor): Input, will be flattened
        """
        lin_vector = torch.linspace(0.1, 5, self.mel_spec_config['n_mels'])
        eye_mat = torch.diag(lin_vector).to(self.device)
        mel_out = self.mel_spec(x.squeeze(1))
        mel_out = torch.tanh(eye_mat @ mel_out)
        return mel_out
        
    
    
    def training_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss'] +\
            self.mel_factor * total_output['stft_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        self.log('Training total loss', total_loss)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value)
        self.log('Validation total loss', total_loss, prog_bar=True)
    
    
if __name__ == "__main__":
    
    vae = Lvl1VQVariationalAutoEncoder(44100, 5.0, 64, 16, 0.001, 0.01)
    