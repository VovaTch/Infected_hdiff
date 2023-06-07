from typing import List, Optional, Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from .base import BaseDiffusionModel
from utils.other import SinusoidalPositionEmbeddings
from utils.diffusion import DiffusionConstants, forward_diffusion_sample, get_index_from_list
from loss import TotalLoss


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.ones((1, 1, 1)))
        self.b = nn.Parameter(torch.zeros((1, 1, 1)))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


def get_emb(sin_inp: torch.Tensor):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1) 
    return torch.flatten(emb, -2, -1)


class DiffusionViT(BaseDiffusionModel):
    """
    Using DiT, this module is used for multiple levels of denoisers, each instance has it's own DiT. 
    TODO: Write the code for the accompanying functions and methods required for the diffusion.
    """
    
    def __init__(self, 
                 in_dim: int,
                 hidden_size: int,
                 token_collect_size: int,
                 num_blocks: int,
                 num_heads: int,
                 num_steps: int,
                 dropout: float=0.0,
                 scheduler: str='cosine',
                 **kwargs) -> pl.LightningModule:
        
        # Initialize variables
        super().__init__(num_steps=num_steps, scheduler=scheduler, **kwargs)
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.token_collect_size = token_collect_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        
        assert hidden_size % num_heads == 0, \
            f'The hidden dimension {hidden_size} must be divisible by the number of heads {num_heads}.'
            
        # Initialize layers
        self.fc_in = nn.Linear(in_dim * token_collect_size, hidden_size)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=num_heads, 
                                                            batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_blocks)
        self.positional_encoding = SinusoidalPositionEmbeddings(self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, in_dim * token_collect_size)
        self.empty_embedding = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        
        # Adaptive layer norm
        self.adaptive_layer_norm = AdaptiveBatchNorm2d(in_dim * token_collect_size)
        
        
    def _patchify(self, x: torch.Tensor):
        """
        Apparently the view function doesn't produce the required result. As such, this function reshapes x in the required patch shape.
        """
        
        # Create the reshaped tensor
        x = self._right_pad_if_necessary(x.transpose(1, 2)).transpose(1, 2)
        x_size = x.size() # BS x C x W, we want BS x C*bl x bl
        x_required_size = [x_size[0], self.token_collect_size * x_size[1], x_size[2] // self.token_collect_size]
        x_reshaped = torch.zeros(x_required_size).to(self.device)
        
        # Fit the data into the required shape
        for block_idx in range(x_required_size[2]):
            # data_slice = x[:, :, block_idx::x_required_size[2]]
            data_slice = x[:, :, block_idx * self.token_collect_size: (block_idx + 1) * self.token_collect_size]
            x_reshaped[:, :, block_idx] = data_slice.transpose(1, 2).flatten(start_dim=1)
            
        # Return the reshaped tensor
        return x_reshaped.transpose(1, 2) # BS x bl x C * W / bl
    
    
    def _depatchify(self, x_reshaped: torch.Tensor):
        """
        After patching, this function reverses the patching process in _patchify. Assumes x_reshaped is BS x C * W / bl x bl
        """
        
        # Create the origin tensor shape
        x_reshaped = x_reshaped.transpose(1, 2)
        x_reshaped_size = x_reshaped.size() # BS x C * W / bl x bl
        x_required_size = [x_reshaped_size[0], 
                           x_reshaped_size[1] // self.token_collect_size, 
                           x_reshaped_size[2] * self.token_collect_size]
        x = torch.zeros(x_required_size).to(self.device)
        
        # Fit the data into the required shape
        for block_idx in range(x_reshaped_size[2]):
            data_slice = x_reshaped[:, :, block_idx]
            intermediate_block = data_slice.view((x_required_size[0], self.token_collect_size, x_required_size[1]))
            x[:, :, block_idx * self.token_collect_size: (block_idx + 1) * self.token_collect_size] =\
                intermediate_block.transpose(1, 2)
            # x[:, :, block_idx::x_reshaped_size[2]] =\
            #     intermediate_block.transpose(1, 2)
                
        # Return the tensor with the original shape
        return x
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        """
        Forward method starts with x: BS x C x W, t: BS
        """
        
        x *= self.data_multiplier
        
        # Transpose and divide the input into chunks
        x = self._patchify(x) # BS x bl x C*W/bl
        num_patches = x.shape[1]
        x_saved = x.clone() # Save for later
        
        # Transpose and device the conditionals into chunks
        if cond is not None:
            cond_list = []
            pos_emb_cond = torch.zeros((x.shape[0], 0, self.hidden_size // self.num_heads)).to(self.device)
            
            # Run over all the conditional and prepare the positional encoding
            for ind_cond in cond:
                
                cond_list.append(self._patchify(ind_cond)) # BS x bl x C*W/bl
                
                # Prepare positional embedding
                pos_emb_range = torch.arange(0, cond_list[-1].shape[1])
                pos_emb_mat = self.positional_encoding(pos_emb_range)
                pos_emb = pos_emb_mat.repeat((x.shape[0], 1, self.num_heads))
                pos_emb_cond = torch.cat((pos_emb_cond, pos_emb), dim=1)
                
            total_cond = torch.cat(cond_list, dim=1) # BS x bl*cond x C*W/bl
            total_cond = self.fc_in(total_cond) + pos_emb_cond # BS x bl*cond x h
        else:
            empty_index = torch.tensor([0 for _ in range(x.shape[0])]).int().to(self.device)
            total_cond = self.empty_embedding(empty_index).unsqueeze(1) # BS x 1 x h
            
        # Prepare timestep embeddings
        t_emb = self.time_mlp(t.int()).unsqueeze(1)
        
        # Prepare inputs to the transformer
        x = self.fc_in(x)
        pos_emb_range = torch.arange(0, x.shape[1]).to(self.device)
        pos_emb_mat = self.positional_encoding(pos_emb_range)
        pos_emb_in = pos_emb_mat.unsqueeze(0).repeat((x.shape[0], 1, 1))
        x += pos_emb_in # BS x bl x h
        
        # Transformer
        x = self.transformer(torch.cat((x, total_cond, t_emb), dim=1), 
                             torch.cat((total_cond, t_emb), dim=1))
        
        # Prepare outputs
        x = self.fc_out(x[:, :num_patches, :]) # BS x bl x C*W/bl
        x = self.adaptive_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self._depatchify(x + x_saved) # BS x C x W
        
        x /= self.data_multiplier
        
        return x
        
        
    def training_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss_combined = self.get_loss(music_slice, time_steps)
        loss_total = loss_combined['total_loss']
        
        for key, value in loss_combined.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        
        return loss_total
    
    
    def validation_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss_combined = self.get_loss(music_slice, time_steps)
        
        for key, value in loss_combined.items():
            if 'loss' in key.split('_'):
                prog_bar = True if key == 'total_loss' else False
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value, prog_bar=prog_bar)
        
        
    def _right_pad_if_necessary(self, x: torch.Tensor):
        """
        Assume x's dimensions are BS x W x C
        """
        
        x_width = x.shape[1]
        if x_width % (self.in_dim * self.token_collect_size) != 0:
            num_missing_values = (self.in_dim * self.token_collect_size)\
                - x_width % (self.in_dim * self.token_collect_size)
            last_dim_padding = (0, 0, 0, num_missing_values)
            x = F.pad(x, last_dim_padding)
        return x
    
    
    def get_loss(self, x_0, t, conditional_list=None) -> Dict:
        
        # Create noisy image
        x_noisy, noise = forward_diffusion_sample(x_0, t, self.diffusion_constants, self.device)
        noise_pred = self(x_noisy, t, conditional_list)
        
        # Predict the image back
        sqrt_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_alphas_cumprod, 
                                                    t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_one_minus_alphas_cumprod, 
                                                              t, x_0.shape)
        x_pred = 1 / sqrt_alphas_cumprod_t * (x_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred)
        
        # Compute losses
        outputs_pred = {'noise_pred': noise_pred, 'output': x_pred}
        outputs_target = {'noise': noise, 'music_slice': x_0}
        
        loss_total = self.loss_obj(outputs_pred, outputs_target)
        return loss_total
    
    
class DiffusionViTSongCond(DiffusionViT):
    """
    This version has embeddings for song conditionals, that can be used for a sort of classifier-free guidance
    """
    
    def __init__(self, 
                 max_num_songs: int = 100, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.song_embeddings = nn.Embedding(num_embeddings=max_num_songs, embedding_dim=self.hidden_size)
        self.song_cond_idx_matching = {}
        
        
    def forward_cond(self, x: torch.Tensor, t: torch.Tensor, song_idx: torch.Tensor=None):
        """
        Expects for song idx a tensor of ints, size BS x Cdim
        """
        
        if song_idx is not None:
            if len(song_idx.size()) == 1:
                song_idx = song_idx.unsqueeze(0)
        
        song_emb = self.song_embeddings(song_idx) if song_idx is not None else None
        
        if song_idx is not None:
            if len(song_emb.size()) == 2:
                song_emb = song_emb.unsqueeze(0)
            
        return self.forward(x, t, cond=song_emb)
            
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        """
        Forward method starts with x: BS x C x W, t: BS
        """
        
        x *= self.data_multiplier
        
        # Transpose and divide the input into chunks
        x = self._patchify(x) # BS x bl x C*W/bl
        num_patches = x.shape[1]
        x_saved = x.clone() # Save for later
        
        # Transpose and device the conditionals into chunks
        if cond is None:
            empty_index = torch.tensor([0 for _ in range(x.shape[0])]).int().to(self.device)
            total_cond = self.empty_embedding(empty_index).unsqueeze(1) # BS x 1 x h
        else:
            total_cond = cond
            
        # Prepare timestep embeddings
        t_emb = self.time_mlp(t.int()).unsqueeze(1)
        
        # Prepare inputs to the transformer
        x = self.fc_in(x)
        pos_emb_range = torch.arange(0, x.shape[1]).to(self.device)
        pos_emb_mat = self.positional_encoding(pos_emb_range)
        pos_emb_in = pos_emb_mat.unsqueeze(0).repeat((x.shape[0], 1, 1))
        x += pos_emb_in # BS x bl x h
        
        # Transformer
        x = self.transformer(torch.cat((x, total_cond, t_emb), dim=1), 
                             torch.cat((total_cond, t_emb), dim=1))
        
        # Prepare outputs
        x = self.fc_out(x[:, :num_patches, :]) # BS x bl x C*W/bl
        x = self.adaptive_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self._depatchify(x + x_saved) # BS x C x W
        
        x /= self.data_multiplier
        
        return x
    
    
    def get_loss(self, x_0, t, song_cond_idx=None) -> Dict:
        
        # Create noisy image
        x_noisy, noise = forward_diffusion_sample(x_0, t, self.diffusion_constants, self.device)
        noise_pred = self.forward_cond(x_noisy, t, song_cond_idx)
        
        # Predict the image back
        sqrt_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_alphas_cumprod, 
                                                    t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_one_minus_alphas_cumprod, 
                                                              t, x_0.shape)
        x_pred = 1 / sqrt_alphas_cumprod_t * (x_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred)
        
        # Compute losses
        outputs_pred = {'noise_pred': noise_pred, 'output': x_pred}
        outputs_target = {'noise': noise, 'music_slice': x_0}
        
        loss_total = self.loss_obj(outputs_pred, outputs_target)
        return loss_total
    
    
    def training_step(self, batch, batch_idx):
        music_slice, track_indices = batch['music slice'], batch['track index']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        track_conditionals = None if random.random() < 0.5 else track_indices
        loss_combined = self.get_loss(music_slice, time_steps, track_conditionals)
        loss_total = loss_combined['total_loss']
        
        for key, value in loss_combined.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        
        return loss_total
    
    
    def validation_step(self, batch, batch_idx):
        music_slice, track_indices = batch['music slice'], batch['track index']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        track_conditionals = None if random.random() < 0.5 else track_indices
        loss_combined = self.get_loss(music_slice, time_steps, track_conditionals)
        
        for key, value in loss_combined.items():
            if 'loss' in key.split('_'):
                prog_bar = True if key == 'total_loss' else False
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value, prog_bar=prog_bar)
                
            
    @torch.no_grad()
    def sample_timestep(self, 
                        x: torch.Tensor, 
                        t: torch.Tensor,
                        conditional_list: Optional[torch.Tensor]=None,
                        guidance_parameter: float=4.0,
                        verbose: bool=False):
        """
        Calls the model to predict the noise in the sound sample and returns 
        the denoised sound sample. 
        Applies noise to this sound sample, if we are not in the last step yet.
        """
        
        betas_t = get_index_from_list(self.diffusion_constants.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.diffusion_constants.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        noise_pred_unguided = self.forward_cond(x, t)
        if conditional_list is not None:
            noise_pred_guided = self.forward_cond(x, t, conditional_list)
            noise_pred = guidance_parameter * noise_pred_guided + (1 - guidance_parameter) * noise_pred_unguided
        else:
            noise_pred = noise_pred_unguided
        
        x = self._right_pad_if_necessary(x.transpose(1, 2)).transpose(1, 2)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.diffusion_constants.posterior_variance, t, x.shape)
        posterior_variance_t[t == 0] = 0
        
        # Show a bunch of plots and prints
        if verbose:
        
            print(sqrt_recip_alphas_t)
            print(betas_t)
            print(sqrt_one_minus_alphas_cumprod_t)
            
            plt.figure(figsize=(25, 5))
            plt.plot((x[0, ...])[0, ...].squeeze(0).cpu().detach().numpy())
            plt.show()
            
            plt.figure(figsize=(25, 5))
            plt.plot((self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
            plt.show()
            
            plt.figure(figsize=(25, 5))
            plt.plot((x[0, ...] - self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
            plt.show()
        
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    
    @torch.no_grad()
    def denoise(self, 
                noisy_input: torch.Tensor, 
                conditionals: Optional[Dict[int, int]]=None, 
                guidance_parameter: float=4.0,
                show_process_plots: bool=False):
        """
        The main denoising method. Expects to get a BS x 1 x Length input, will output a denoised music sample.

        Args:
            noisy_input (torch.Tensor): the input, can be noise, can be noisy music. The model should handle both.
        """
        
        multiplied_noisy_input = self.data_multiplier * noisy_input
        
        running_slice = multiplied_noisy_input.clone()
        batch_size = noisy_input.shape[0]
        for time_step in reversed(range(self.num_steps)):

            # If using the denoise method, we don't want to use the data multiplier again.
            temp = self.data_multiplier
            self.data_multiplier = 1.0

            time_input = torch.tensor([time_step for _ in range(batch_size)]).to(self.device)
            if conditionals is not None:
                step_conditionals = torch.tensor([song_idx for song_idx, activation_idx in conditionals.items() 
                                                  if activation_idx >= time_step]).to(noisy_input.device)
            else:
                step_conditionals = None
                
            running_slice = self.sample_timestep(running_slice, time_input, step_conditionals, guidance_parameter)
            
            # Returning the data multiplier to its original value
            self.data_multiplier = temp
            
            if show_process_plots:
                plt.figure(figsize=(25, 5))
                plt.ylim((-1.1, 1.1))
                plt.plot(running_slice[0, ...].squeeze(0).cpu().detach().numpy())
                plt.show()
                    
        return running_slice / self.data_multiplier