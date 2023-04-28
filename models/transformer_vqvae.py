from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_codebook import VQCodebook
from .base import BaseNetwork
from loss import TotalLoss
from utils.other import SinusoidalPositionEmbeddings


def _patchify(x: torch.Tensor, patch_collection_size: int):
    """
    Dimensions in: BS x W x C. Dimensions out: BS x W / P x C * P
    """
    
    x_size = x.shape # BS x C x W
    x = x.view((x_size[0], -1, x_size[2] * patch_collection_size))
    return x
    
    
def _depatchify(x: torch.Tensor, patch_collection_size: int):
    """
    Dimensions in: BS x W x C, Dimensions out: BS x W * P x C / P
    """
    
    x_size = x.shape # BS x W x C
    x = x.view((x_size[0], -1, x_size[2] // patch_collection_size))
    return x
    
    
def _right_pad_if_necessary(x: torch.Tensor, patch_collection_size: int):
    """
    Dimensions in: BS x C x W, adds W such that it matches the patch collection size.
    """
    
    x_size = x.shape
    x_in = x.clone()
    
    if x_size[1] * x_size[2] % patch_collection_size == 0:
        return x_in
    
    x_in = F.pad(x_in, (0, patch_collection_size - (x_size[1] * x_size[2] // patch_collection_size)))
    return x_in


class TransEncoder(nn.Module):

    def __init__(self, 
                 dim_changes_list: List[int], 
                 input_channels: int = 1, 
                 output_channels: int = 8,
                 hidden_size: int = 256,
                 patch_collection_size: int = 32,
                 n_heads: int = 8,
                 n_trans: int = 1):
        
        super().__init__()
        
        self.dim_changes_list = dim_changes_list
        self.patch_collection_size = patch_collection_size
        
        # Transformer layer
        trans_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dropout=0.0, nhead=n_heads, norm_first=True)
        
        # Define transformers
        self.attn_stack = nn.ModuleList([
            nn.TransformerEncoder(trans_layer, num_layers=n_trans) for _ in range(len(dim_changes_list) + 1)
        ])
        
        # Define fully connected layers
        self.fc_stack = nn.ModuleList([nn.Linear(patch_collection_size * input_channels, hidden_size)] + [
            nn.Linear(hidden_size * dim_change, hidden_size) for dim_change in dim_changes_list
        ] + [nn.Linear(hidden_size // patch_collection_size * dim_changes_list[-1], output_channels)])
        
        
    def forward(self, x: torch.Tensor):
        
        # Patchify
        x = x.permute((0, 2, 1))
        x = _patchify(x, self.patch_collection_size)
        
        # Run through the stack of fully connected to transformers
        for (trans_module, fc_module, dim_change) in zip(self.attn_stack, self.fc_stack[:-1], self.dim_changes_list):
            x = fc_module(x)
            x = F.gelu(x)
            x = trans_module(x)
            x = _patchify(x, dim_change)
            
        # End linear layer
        x = _depatchify(x, self.patch_collection_size)
        x = self.fc_stack[-1](x)
        return x
            
        
class TransDecoder(nn.Module):

    def __init__(self, 
                 dim_changes_list: List[int], 
                 input_channels: int = 1, 
                 output_channels: int = 8,
                 hidden_size: int = 256,
                 patch_collection_size: int = 32,
                 n_heads: int = 8,
                 n_trans: int = 1):
        
        super().__init__()
        
        self.dim_changes_list = dim_changes_list
        self.patch_collection_size = patch_collection_size
        
        # Transformer layer
        trans_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dropout=0.0, nhead=n_heads, norm_first=True)

        # Define transformers
        self.attn_stack = nn.ModuleList([
            nn.TransformerEncoder(trans_layer, num_layers=n_trans) for _ in range(len(dim_changes_list) + 1)
        ])
        
        # Define fully connected layers
        self.fc_stack = nn.ModuleList([nn.Linear(output_channels * patch_collection_size, hidden_size)] + [
            nn.Linear(hidden_size // dim_change, hidden_size) for dim_change in dim_changes_list
        ] + [nn.Linear(hidden_size // patch_collection_size // dim_changes_list[-1], input_channels)])


    def forward(self, x: torch.Tensor):
        
        # Depatchify
        x = _patchify(x, self.patch_collection_size)
        
        # Run the stacks of fully connected and transformers
        for (trans_module, fc_module, dim_change) in zip(self.attn_stack, self.fc_stack[:-1], self.dim_changes_list):
            x = fc_module(x)
            x = F.gelu(x)
            x = trans_module(x)
            x = _depatchify(x, dim_change)
            
        # End linear layer
        x = _depatchify(x, self.patch_collection_size)
        x = self.fc_stack[-1](x)
        x = x.permute((0, 2, 1))
        return x


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


class TransformerVQVAE(BaseNetwork):
    """
    Experimental transformer-based VQVAE network, 
    might work better than the 1d conv vqvae as it takes the entire context and is more scalable.
    """
    
    def __init__(self, 
                 learning_rate: float, 
                 weight_decay: float, 
                 batch_size: int, 
                 epochs: int, 
                 scheduler_type: str, 
                 latent_depth: int,
                 num_transformer_layers_per_stage: int = 1,
                 loss_obj: TotalLoss = None,
                 steps_per_epoch: int = 500,
                 vocabulary_size: int = 8192,
                 input_channels: int = 1,
                 hidden_size: int = 256,
                 patch_collection_size: int = 32,
                 n_heads: int = 8,
                 channel_dim_change_list: List[int] = [2, 2, 2, 2, 2, 2],
                 **kwargs):
        
        super().__init__(learning_rate, weight_decay, batch_size, epochs, scheduler_type, steps_per_epoch, **kwargs)
        
        # Initiate variables
        self.latent_depth = latent_depth
        self.vocabulary_size = vocabulary_size
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.patch_collection_size = patch_collection_size
        self.channel_dim_change_list = channel_dim_change_list
        self.loss_obj = loss_obj
        self.n_trans = num_transformer_layers_per_stage
        self.n_heads = n_heads
        
        # Initiate encoder and decoder change dim lists
        encoder_dim_changes = channel_dim_change_list
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        
        # Initiate encoder and decoder
        self.encoder = TransEncoder(encoder_dim_changes, 
                                    input_channels=input_channels,
                                    hidden_size=hidden_size,
                                    patch_collection_size=patch_collection_size,
                                    n_heads=n_heads,
                                    n_trans=num_transformer_layers_per_stage)
        self.decoder = TransDecoder(decoder_dim_changes,
                                    input_channels=input_channels,
                                    hidden_size=hidden_size,
                                    patch_collection_size=patch_collection_size,
                                    n_heads=n_heads,
                                    n_trans=num_transformer_layers_per_stage)
        self.vq_module = VQ1D(latent_depth, num_tokens=vocabulary_size)
        
        
    def forward(self, x: torch.Tensor):
        """
        Runs the net through the encoder, the vqvae codebook, and the decoder.
        """
        
        # Pad if necessary
        x_padded = _right_pad_if_necessary(x, self.patch_collection_size) # BS x C x W
        
        # Run through encoder, to decoder
        z_e = self.encoder(x_padded).permute((0, 2, 1))
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'].permute((0, 2, 1)))
        
        total_output = {**vq_block_output,
                        'output': x_out}
        
        loss_target = {'music_slice': x_padded,
                       'z_e': z_e}
        
        # Collect losses
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
        """
        Randomly shuffles the codebook to encourage exploration of its entirety.
        """
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