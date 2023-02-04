from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from .vq_codebook import VQCodebook
import loaders

DATASETS = {'music_slice_dataset': loaders.MP3SliceDataset}

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
            
        return torch.tanh(x_out)


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
        


class Lvl1VQVariationalAutoEncoder(pl.LightningModule):
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
                 channel_dim_change_list: List[int] = [2, 2, 2, 4, 4],
                 dataset_name: str='music_slice_dataset',
                 dataset_path: str='data/music_samples',
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
        encoder_channel_list = [hidden_size, hidden_size * 2, hidden_size * 4, hidden_size * 8, hidden_size * 16, latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = list(reversed(encoder_channel_list))
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        sin_locations = None
        
        # Initialize network parts
        self.encoder = Lvl1Encoder(encoder_channel_list, encoder_dim_changes)
        self.decoder = Lvl1Decoder(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations)
        self.vq_module = Lvl1VQ(latent_depth, num_tokens=vocabulary_size)
        
        # Datasets
        assert dataset_name in DATASETS, f'Dataset {dataset_name} is not in the datasets options.'
        assert 0 <= eval_split_factor <= 1, f'The split factor must be between 0 and 1, current value is {eval_split_factor}'
        self.dataset = None
        self.eval_split_factor = eval_split_factor
        self.dataset_name = dataset_name
        
        # Optimizers
        assert optimizer_name in ['none', 'one_cycle_lr', 'reduce_on_platou'] # TODO fix typo, program the schedulers in
        
        
    def forward(self, x, extract_losses: bool=False):
        
        z_e = self.encoder(x)
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'])
        
        total_output = {**vq_block_output,
                        'output': x_out}
        
        if extract_losses:
            total_output.update({'reconstruction_loss': F.mse_loss(x, x_out)})
        
        return total_output
    
    
    def train_dataloader(self):
        self._set_dataset()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        self._set_dataset()
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def configure_optimizers(self):
        self._set_dataset()
        if len(self.dataset) % self.batch_size == 0:
            total_steps = len(self.dataset) // self.batch_size
        else:
            total_steps = len(self.dataset) // self.batch_size + 1
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                                        epochs=self.epochs,
                                                        steps_per_epoch=total_steps)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1e6)
        scheduler_settings= {'scheduler': scheduler,
                             'interval': 'step',
                             'monitor': 'Training reconstruction loss',
                             'frequency': 1} # Set the scheduler to step every batch
        
        return [optimizer], [scheduler_settings]
    
    
    def training_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
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
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value)
        self.log('Validation total loss', total_loss)
        
    def _set_dataset(self):
        
        if self.dataset is None:
            self.dataset = DATASETS[self.dataset_name](**self.cfg, audio_dir=self.dataset_path, slice_time=self.slice_length)
            train_dataset_length = int(len(self.dataset) * (1 - self.eval_split_factor))
            self.train_dataset, self.eval_dataset = random_split(self.dataset, 
                                                                (train_dataset_length, 
                                                                len(self.dataset) - train_dataset_length))
    
    
if __name__ == "__main__":
    
    vae = Lvl1VQVariationalAutoEncoder(44100, 5.0, 64, 16, 0.001, 0.01)
    