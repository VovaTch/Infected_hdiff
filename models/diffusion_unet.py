from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffusionModel
from utils.other import SinusoidalPositionEmbeddings
from utils.diffusion import forward_diffusion_sample, get_index_from_list


class WaveNetDiffusion(BaseDiffusionModel):
    
    def __init__(self, 
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 filter_size_encoder: int, 
                 filter_size_decoder: int,
                 num_input_channels: int=1,
                 num_filters: int=1,
                 time_emb_dim: int=32,
                 slice_length: int=512,
                 **kwargs):
            
        super().__init__(**kwargs)
        
        # Initialize variables
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.filter_size_encoder = filter_size_encoder
        self.filter_size_decoder = filter_size_decoder
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.cfg = kwargs
        
        # Initialize channel lists
        enc_channel_in = [self.num_input_channels] + [min(self.num_decoder_layers, (i + 1)) * self.num_filters 
                                                      for i in range(self.num_encoder_layers - 1)]
        enc_channel_out = [min(self.num_decoder_layers, (i + 1)) * self.num_filters for i in range(self.num_encoder_layers)]
        dec_channel_out = enc_channel_out[:self.num_decoder_layers][::-1]
        dec_channel_in = [enc_channel_out[-1] * 2] + [enc_channel_out[- i - 1] + dec_channel_out[i - 1] 
                                                                         for i in range(1, self.num_decoder_layers)]
        
        # Initialize encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder_time_emb = nn.ModuleList()
        self.decoder_time_emb = nn.ModuleList()
        
        for i in range(self.num_encoder_layers):
            self.encoder.append(nn.Conv1d(enc_channel_in[i], enc_channel_out[i], self.filter_size_encoder, 
                                          padding=self.filter_size_encoder // 2))
            
        for i in range(self.num_encoder_layers):
            self.encoder_time_emb.append(nn.Sequential(
                nn.Linear(time_emb_dim, enc_channel_out[i]),
                nn.GELU(),
            ))

        for i in range(self.num_decoder_layers):
            self.decoder.append(nn.Conv1d(dec_channel_in[i], dec_channel_out[i], self.filter_size_decoder,
                                          padding=self.filter_size_decoder // 2))
            
        for i in range(self.num_encoder_layers):
            self.decoder_time_emb.append(nn.Sequential(
                nn.Linear(time_emb_dim, dec_channel_out[i]),
                nn.GELU(),
            ))

        self.middle_layer = nn.Sequential(
            nn.Conv1d(enc_channel_out[-1], enc_channel_out[-1], self.filter_size_encoder, 
                      padding=self.filter_size_encoder // 2),
            nn.LeakyReLU(0.2)
        )
        
        output_layer_in_size = enc_channel_in[1] + self.num_input_channels
        self.output_layer = nn.Sequential(
            nn.Conv1d(output_layer_in_size, 
                      self.num_input_channels, kernel_size=1), # Maybe it's reverse
            nn.Tanh()
        )
        
        # Time embedding
        self.time_emb = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
            )
        
        # Ending conv
        
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        
        encoder = list()
        input = x

        # Downsampling
        for i in range(self.num_encoder_layers):
            
            # time embedding
            t_emb = self.time_emb(t)
            t_emb = self.encoder_time_emb[i](t_emb)
            t_emb = t_emb[(..., ) + (None, )]
            
            # Forward for x
            x = self.encoder[i](x)
            x += t_emb
            x = F.leaky_relu(x, 0.2)
            encoder.append(x)
            x = x[:, :, ::2]

        x = self.middle_layer(x)

        # Upsampling
        for i in range(self.num_decoder_layers):
            
            # time embedding
            t_emb = self.time_emb(t)
            t_emb = self.decoder_time_emb[i](t_emb)
            t_emb = t_emb[(..., ) + (None, )]
            
            # Forward for x
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='linear', align_corners=True)
            x = self._crop_and_concat(x, encoder[self.num_encoder_layers - i - 1])
            x = self.decoder[i](x)
            x += t_emb
            x = F.leaky_relu(x, 0.2)

        # Concat with original input
        x = self._crop_and_concat(x, input)
        x = self.output_layer(x)
        
        return x
        
        
    def _right_pad_if_necessary(self, x: torch.Tensor):
        """
        Pad x if necessary; can also make it a transparent method if needed
        """
        return x
    
    
    def _crop_and_concat(self, x1: torch.Tensor, x2: torch.Tensor):
        if x2.shape[-1] > x1.shape[-1]: # This probably should be deleted if anyone reads this besides me.
            crop_x2 = self._crop(x2, x1.shape[-1])
        elif x1.shape[-1] > x2.shape[-1]:
            crop_x2 = self._crop(x1, x2.shape[-1])
        else:
            crop_x2 = x2
        x = torch.cat([x1, crop_x2], dim=1)
        return x


    def _crop(self, tensor: torch.Tensor, target_shape):
        # Center crop
        shape = tensor.shape[-1]
        diff = shape - target_shape
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:,:,crop_start:-crop_end]
    
    
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