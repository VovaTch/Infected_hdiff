from typing import List, Optional, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt

from .base import BaseNetwork
from utils.diffusion import DiffusionConstants, forward_diffusion_sample, get_index_from_list
from .dit_facebook import DiTBlock, TimestepEmbedder


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


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class DiffusionViT(BaseNetwork):
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
                 loss_dict: Dict[str, float],
                 dropout: float=0.0,
                 scheduler: str='cosine',
                 **kwargs) -> pl.LightningModule:
        
        # Initialize variables
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.token_collect_size = token_collect_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_steps = num_steps
        self.diffusion_constants = DiffusionConstants(self.num_steps, scheduler=scheduler)
        self.mel_factor = loss_dict['loss_mel']
        self.mel_factor_sub_1 = loss_dict['loss_mel_sub_1']
        self.mel_factor_sub_2 = loss_dict['loss_mel_sub_2']
        self.reconstruction_factor = loss_dict['loss_reconstruction']
        assert hidden_size % num_heads == 0, \
            f'The hidden dimension {hidden_size} must be divisible by the number of heads {num_heads}.'
            
        # Initialize mel spectrogram, TODO: Might do multiple ones for multiple losses
        self.mel_spec = None
        if 'mel_spec_config' in kwargs:
            self.mel_spec_config = kwargs['mel_spec_config']
            self.mel_spec = MelSpectrogram(sample_rate=kwargs['sample_rate'], **self.mel_spec_config)
            
        # Initialize the sub mel specs for additional losses
        self.mel_spec_sub_1 = None
        if 'mel_spec_sub_1_config' in kwargs:
            self.mel_spec_config_sub_1 = kwargs['mel_spec_sub_1_config']
            self.mel_spec_sub_1 = MelSpectrogram(sample_rate=kwargs['sample_rate'], **self.mel_spec_config_sub_1)
            
        # Initialize the sub mel specs for additional losses
        self.mel_spec_sub_2 = None
        if 'mel_spec_sub_2_config' in kwargs:
            self.mel_spec_config_sub_2 = kwargs['mel_spec_sub_2_config']
            self.mel_spec_sub_2 = MelSpectrogram(sample_rate=kwargs['sample_rate'], **self.mel_spec_config_sub_2)
            
        # Initialize layers
        self.fc_in = nn.Linear(in_dim * token_collect_size, hidden_size)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=num_heads, 
                                                            batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_blocks)
        self.positional_encoding = SinusoidalPositionEmbeddings(self.hidden_size // num_heads)
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
            data_slice = x[:, :, block_idx::x_required_size[2]]
            #data_slice = x[:, :, block_idx * self.token_collect_size: (block_idx + 1) * self.token_collect_size]
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
            # x[:, :, block_idx * self.token_collect_size: (block_idx + 1) * self.token_collect_size] =\
            #     intermediate_block.transpose(1, 2)
            x[:, :, block_idx::x_reshaped_size[2]] =\
                intermediate_block.transpose(1, 2)
                
        # Return the tensor with the original shape
        return x
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        """
        Forward method starts with x: BS x C x W, t: BS
        """
        
        # Prepare positional embeddings
        pos_emb_range = torch.arange(0, x.shape[2] // self.token_collect_size).to(self.device)
        pos_emb_mat = self.positional_encoding(pos_emb_range)
        pos_emb_in = pos_emb_mat.unsqueeze(0).repeat((x.shape[0], 1, self.num_heads))
        
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
        x = self.fc_in(x) + pos_emb_in # BS x bl x h
        
        # Transformer
        x = self.transformer(torch.cat((x, total_cond, t_emb), dim=1), 
                             torch.cat((total_cond, t_emb), dim=1))
        
        # Prepare outputs
        x = self.fc_out(x[:, :num_patches, :]) # BS x bl x C*W/bl
        x = self.adaptive_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self._depatchify(x + x_saved) # BS x C x W
        
        return x
        
        
    def training_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss = self.get_loss(music_slice, time_steps)
        loss_total = loss['diffusion_error_loss'] * self.reconstruction_factor +\
                     loss['stft_loss'] * self.mel_factor +\
                     loss['stft_loss_sub_1'] * self.mel_factor_sub_1 +\
                     loss['stft_loss_sub_2'] * self.mel_factor_sub_2
        
        loss_total = loss['diffusion_error_loss'] + loss['stft_loss'] * self.mel_factor\
            + loss['stft_loss_sub_1'] * self.mel_factor_sub_1 + loss['stft_loss_sub_2'] * self.mel_factor_sub_2
        self.log('Training diffusion loss', loss['diffusion_error_loss'])
        self.log('Training stft loss', loss['stft_loss'])
        self.log('Training stft loss sub 1', loss['stft_loss_sub_1'])
        self.log('Training stft loss sub 2', loss['stft_loss_sub_2'])
        self.log('Training total loss', loss_total)
        return loss_total
    
    
    def validation_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss = self.get_loss(music_slice, time_steps)
        loss_total = loss['diffusion_error_loss'] * self.reconstruction_factor +\
                     loss['stft_loss'] * self.mel_factor +\
                     loss['stft_loss_sub_1'] * self.mel_factor_sub_1 +\
                     loss['stft_loss_sub_2'] * self.mel_factor_sub_2
                     
        self.log('Validation diffusion loss', loss['diffusion_error_loss'])
        self.log('Validation stft loss', loss['stft_loss'])
        self.log('Validation stft loss sub 1', loss['stft_loss_sub_1'])
        self.log('Validation stft loss sub 2', loss['stft_loss_sub_2'])
        self.log('Validation total loss', loss_total, prog_bar=True)
        
        
    def _right_pad_if_necessary(self, x: torch.Tensor):
        """
        Assume x's dimensions are BS x W x C
        """
        
        x_width = x.shape[1]
        if x_width % (self.in_dim * self.token_collect_size) != 0:
            num_missing_values = (self.in_dim * self.token_collect_size)\
                - x_width % (self.in_dim * self.token_collect_size)
            last_dim_padding = (0, num_missing_values)
            x = F.pad(x, last_dim_padding)
        return x
    
    
    def get_loss(self, x_0, t, conditional_list=None):
        
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
        stft_loss = F.l1_loss(self._mel_spec_and_process(x_pred), 
                              self._mel_spec_and_process(x_0))
        stft_loss_sub_1 = F.l1_loss(self._mel_spec_sub_1_and_process(x_pred),
                                    self._mel_spec_sub_1_and_process(x_0))
        stft_loss_sub_2 = F.l1_loss(self._mel_spec_sub_2_and_process(x_pred),
                                    self._mel_spec_sub_2_and_process(x_0))
        return {'diffusion_error_loss': F.l1_loss(noise, noise_pred),
                'stft_loss': stft_loss,
                'stft_loss_sub_1': stft_loss_sub_1,
                'stft_loss_sub_2': stft_loss_sub_2}
        
        
    def _mel_spec_and_process(self, x: torch.Tensor):
        """
        To prepare the mel spectrogram loss, everything needs to be prepared.

        Args:
            x (torch.Tensor): Input, will be flattened
        """
        lin_vector = torch.linspace(0.5, 10, self.mel_spec_config['n_mels'])
        eye_mat = torch.diag(lin_vector).to(self.device)
        mel_out = self.mel_spec(x.flatten(start_dim=0, end_dim=1))
        mel_out = torch.log(eye_mat @ mel_out + 1e-5)
        return mel_out
    
    
    def _mel_spec_sub_1_and_process(self, x: torch.Tensor):
        '''
        Prepares the sub_1 mel spectrogram loss
        '''
        
        lin_vector = torch.linspace(1, 1, self.mel_spec_config_sub_1['n_mels'])
        eye_mat = torch.diag(lin_vector).to(self.device)
        mel_out = self.mel_spec_sub_1(x.flatten(start_dim=0, end_dim=1))
        mel_out = torch.log(eye_mat @ mel_out + 1e-5)
        return mel_out
    
    
    def _mel_spec_sub_2_and_process(self, x: torch.Tensor):
        '''
        Prepares the sub_2 mel spectrogram loss
        '''
        
        lin_vector = torch.linspace(0.1, 20, self.mel_spec_config_sub_2['n_mels'])
        eye_mat = torch.diag(lin_vector).to(self.device)
        mel_out = self.mel_spec_sub_2(x.flatten(start_dim=0, end_dim=1))
        mel_out = torch.log(eye_mat @ mel_out + 1e-5)
        return mel_out
    
    
    @torch.no_grad()
    def sample_timestep(self, 
                        x: torch.Tensor, 
                        t: torch.Tensor,
                        conditional_list: Optional[List[torch.Tensor]]=None):
        """
        Calls the model to predict the noise in the sound sample and returns 
        the denoised sound sample. 
        Applies noise to this sound sample, if we are not in the last step yet.
        """
        
        betas_t = get_index_from_list(self.diffusion_constants.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.diffusion_constants.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.diffusion_constants.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (x - 
                                            betas_t * self(x, t, conditional_list) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.diffusion_constants.posterior_variance, t, x.shape)
        posterior_variance_t[t == 0] = 0
        
        # print(sqrt_recip_alphas_t)
        # print(betas_t)
        # print(sqrt_one_minus_alphas_cumprod_t)
        
        # plt.figure(figsize=(25, 5))
        # plt.plot((x[0, ...])[0, ...].squeeze(0).cpu().detach().numpy())
        # plt.show()
        
        # plt.figure(figsize=(25, 5))
        # plt.plot((self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
        # plt.show()
        
        # plt.figure(figsize=(25, 5))
        # plt.plot((x[0, ...] - self(x, t, conditional_list)[0, ...]).squeeze(0).cpu().detach().numpy())
        # plt.show()
        
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
        
    @torch.no_grad()
    def denoise(self, noisy_input: torch.Tensor, conditionals: Optional[List[torch.Tensor]]=None, show_process_plots: bool=False):
        """
        The main denoising method. Expects to get a BS x 1 x Length input, will output a denoised music sample.

        Args:
            noisy_input (torch.Tensor): the input, can be noise, can be noisy music. The model should handle both.
        """
        
        running_slice = noisy_input.clone()
        batch_size = noisy_input.shape[0]
        for time_step in reversed(range(self.num_steps)):
            
            if time_step < 30:
            
                time_input = torch.tensor([time_step for _ in range(batch_size)]).to(self.device)
                running_slice = self.sample_timestep(running_slice, time_input, conditionals)
                running_slice = torch.tanh(running_slice)
                
                if show_process_plots:
                    plt.figure(figsize=(25, 5))
                    plt.ylim((-1.1, 1.1))
                    plt.plot(running_slice[0, ...].squeeze(0).cpu().detach().numpy())
                    plt.show()
            
        return running_slice