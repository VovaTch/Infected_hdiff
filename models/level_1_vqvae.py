from typing import Any, List

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
        
        return x
    

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


class Lvl1VQ(nn.Module):
    
    
    def __init__(self, token_dim, num_tokens: int=512):
        
        super().__init__()
        self.vq_codebook = VQCodebook(token_dim, num_tokens=num_tokens)
        
        
    def forward(self, z_e, extract_losses: bool=False):
        
        indices = self.vq_codebook.extract_indices(z_e)
        z_q = self.vq_codebook(indices)
        output = {'indices': indices, 'v_q': z_q}
        
        if extract_losses:
            losses = {'alignment_loss': (z_e.detach() - z_q) ** 2,
                      'commitment_loss': (z_e - z_q.detach()) ** 2}
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
                 dataset_name: str='music_slice_dataset',
                 optimizer_name: str='one_cycle_lr',
                 eval_split_factor: float=0.1,
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
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta_factor = beta_factor
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size, hidden_size * 2, hidden_size * 4, hidden_size * 8, hidden_size * 16, latent_depth]
        encoder_dim_changes = [5, 5, 5, 5, 5]
        decoder_channel_list = encoder_channel_list.reverse()
        decoder_dim_changes = [5, 5, 5, 5, 5]
        
        # Initialize network parts
        self.encoder = Lvl1Encoder(encoder_channel_list, encoder_dim_changes)
        self.decoder = Lvl1Decoder(decoder_channel_list, decoder_dim_changes)
        self.vq_module = Lvl1VQ(latent_depth)
        
        # Datasets
        assert dataset_name in DATASETS, f'Dataset {dataset_name} is not in the datasets options.'
        assert 0 <= eval_split_factor <= 1, f'The split factor must be between 0 and 1, current value is {eval_split_factor}'
        self.dataset = DATASETS[dataset_name](**kwargs)
        train_dataset_length = int(len(self.dataset) * (1 - eval_split_factor))
        self.train_dataset, self.eval_dataset = random_split(self.dataset, 
                                                             (train_dataset_length, 
                                                              len(self.dataset) - train_dataset_length))
        
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=True)
    
    
    def configure_optimizers(self):
        
        if len(self.dataset) % self.batch_size == 0:
            total_steps = len(self.dataset) // self.batch_size
        else:
            total_steps = len(self.dataset) // self.batch_size + 1
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                                        epochs=self.epochs,
                                                        steps_per_epoch=total_steps)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        
        total_output = self.forward(batch, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        self.log('Training total loss', total_loss)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        
        total_output = self.forward(batch, extract_losses=True)
        total_loss = total_output['reconstruction_loss'] + total_output['alignment_loss'] +\
            self.beta_factor * total_output['commitment_loss']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value)
        self.log('Validation total loss', total_loss)
    
    
    
if __name__ == "__main__":
    
    vae = Lvl1VQVariationalAutoEncoder(44100, 5.0, 64, 16, 0.001, 0.01)
    