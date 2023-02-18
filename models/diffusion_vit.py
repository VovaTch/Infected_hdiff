from typing import List, Optional

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


def get_emb(sin_inp: torch.Tensor):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1) 
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


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
                 mel_factor: float=100.0,
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
        self.mel_factor = mel_factor
        assert hidden_size % num_heads == 0, \
            f'The hidden dimension {hidden_size} must be divisible by the number of heads {num_heads}.'
            
        # Initialize mel spectrogram, TODO: Might do multiple ones for multiple losses
        self.mel_spec = None
        if 'mel_spec_config' in kwargs:
            self.mel_spec_config = kwargs['mel_spec_config']
            self.mel_spec = MelSpectrogram(sample_rate=kwargs['sample_rate'], **self.mel_spec_config)
            
        
        # Initialize layers
        self.fc_in = nn.Linear(in_dim * token_collect_size, hidden_size)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=num_heads, 
                                                            batch_first=True, dropout=dropout, norm_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_blocks)
        self.positional_encoding = PositionalEncoding1D(self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, in_dim * token_collect_size)
        self.empty_embedding = nn.Embedding(num_embeddings=1, embedding_dim=hidden_size)
        self.timestep_embedding = nn.Embedding(num_embeddings=num_steps, embedding_dim=hidden_size)
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        """
        Forward method starts with x: BS x C x W, t: BS
        """
        
        # Transpose and divide the input into chunks
        x = x.transpose(1, 2) # BS x W x C
        in_shape = x.shape
        x = self._right_pad_if_necessary(x) # BS x W+ x C
        x = x.view((x.shape[0], -1, self.in_dim * self.token_collect_size)) # BS x W+/bl * C*bl
        
        # Transpose and device the conditionals into chunks
        if cond is not None:
            cond_list = []
            for ind_cond in cond:
                cond_list.append(torch.zeros_like(ind_cond))
                cond_list[-1] = ind_cond.transpose(1, 2) # BS x W x C
                cond_list[-1] = self._right_pad_if_necessary(cond_list[-1]) # BS x W+ x C
                cond_list[-1] = cond_list[-1].view((x.shape[0], -1, self.in_dim * self.token_collect_size)) # BS x W+/bl x C*bl
            total_cond = torch.cat(cond_list, dim=1) # BS x Cond*W+/bl * C*bl
            total_cond = self.fc_in(total_cond) # BS x Cond*W+/bl x h
        else:
            empty_index = torch.tensor([0 for _ in range(x.shape[0])]).int().to(self.device)
            total_cond = self.empty_embedding(empty_index).unsqueeze(1) # BS x 1 x h
            
        # Prepare timestep embeddings
        t_emb = self.timestep_embedding(t.int()).unsqueeze(1)
        
        # Prepare inputs to the transformer
        x = self.fc_in(x) # BS x W+/bl x h
        x += self.positional_encoding(x)
        total_cond += self.positional_encoding(total_cond)
        
        # Transformer
        x = self.transformer(torch.cat((x, t_emb), dim=1), 
                             torch.cat((total_cond, t_emb), dim=1))
        
        # Prepare outputs
        x = self.fc_out(x[:, :-1, :]) # BS x W+/bl x C*bl
        x = x.view(in_shape) # BS x W x C
        
        return x.transpose(1, 2)
        
        
    def training_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss = self.get_loss(music_slice, time_steps)
        loss_total = loss['diffusion_error_loss'] + loss['stft_loss'] * self.mel_factor
        self.log('Training diffusion loss', loss['diffusion_error_loss'])
        self.log('Training stft loss', loss['stft_loss'])
        self.log('Training total loss', loss_total)
        return loss_total
    
    
    def validation_step(self, batch, batch_idx):
        music_slice = batch['music slice']
        time_steps = torch.randint(1, self.num_steps, (music_slice.shape[0],)).to(self.device)
        loss = self.get_loss(music_slice, time_steps)
        loss_total = loss['diffusion_error_loss'] + loss['stft_loss'] * self.mel_factor
        self.log('Validation diffusion loss', loss['diffusion_error_loss'])
        self.log('Validation stft loss', loss['stft_loss'])
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
        stft_loss = F.l1_loss(self._mel_spec_and_process(torch.tanh(x_pred)), 
                              self._mel_spec_and_process(torch.tanh(x_0)))
        return {'diffusion_error_loss': F.l1_loss(noise, noise_pred),
                'stft_loss': stft_loss}
        
        
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
    def denoise(self, noisy_input: torch.Tensor, conditionals: Optional[List[torch.Tensor]]=None):
        """
        The main denoising method. Expects to get a BS x 1 x Length input, will output a denoised music sample.

        Args:
            noisy_input (torch.Tensor): the input, can be noise, can be noisy music. The model should handle both.
        """
        
        running_slice = noisy_input.clone()
        batch_size = noisy_input.shape[0]
        for time_step in reversed(range(self.num_steps)):
            
            time_input = torch.tensor([time_step for _ in range(batch_size)]).to(self.device)
            running_slice = self.sample_timestep(running_slice, time_input, conditionals)
            running_slice = torch.tanh(running_slice)
        
        #running_slice[running_slice < -1] = -1
        #running_slice[running_slice > 1] = 1
        return running_slice