from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffusionModel
from utils.other import SinusoidalPositionEmbeddings


class WaveNetDiffusion(BaseDiffusionModel):
    
    def __init__(self, 
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 filter_size_encoder: int, 
                 filter_size_decoder: int,
                 num_input_channels: int=1,
                 num_filters: int=1,
                 time_emb_dim: int=32,
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
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.num_input_channels, self.num_input_channels, kernel_size=1),
            nn.Tanh()
        )
        
        # Time embedding
        self.time_emb = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
            )
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]=None):
        
        encoder = list()
        input = x

        # Downsampling
        for i in range(self.num_encoder_layers):
            
            x = self.encoder[i](x)
            x = F.leaky_relu(x, 0.2)
            encoder.append(x)
            x = x[:, :, ::2]

        x = self.middle_layer(x)

        # Upsampling
        for i in range(self.num_decoder_layers):
            x = F.interpolate(x, size=x.shape[-1] * 2, mode='linear', align_corners=True)
            x = self._crop_and_concat(x, encoder[self.num_encoder_layers - i - 1])
            x = self.decoder[i](x)
            x = F.leaky_relu(x, 0.2)

        # Concat with original input
        x = self._crop_and_concat(x, input)
        
        return torch.sum(x, dim=1).unsqueeze(1)
        
        
    def _right_pad_if_necessary(self, x: torch.Tensor):
        """
        Pad x if necessary; can also make it a transparent method if needed
        """
        return x
    
    
    def _crop_and_concat(self, x1: torch.Tensor, x2: torch.Tensor):
        if x2.shape[-1] != x1.shape[-1]: # This probably should be deleted if anyone reads this besides me.
            crop_x2 = self._crop(x2, x1.shape[-1])
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