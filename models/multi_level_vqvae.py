from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from .vq_codebook import VQCodebook
from loaders import MP3SliceDataset, Lvl2InputDataset
from .base import BaseNetwork

DATASETS = {'music_slice_dataset': MP3SliceDataset,
            'lvl2_dataset': Lvl2InputDataset}


class ConvDownsample(nn.Module):
    '''
    A small module handling downsampling via a convolutional layer instead of e.g. Maxpool.
    '''
    
    def __init__(self, kernel_size: int, downsample_divide: int, in_dim: int):
        
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample_divide = downsample_divide
        self.in_dim = in_dim
        self.padding_needed = (kernel_size - 2) + (downsample_divide - 2)
        self.padding_needed = 0 if self.padding_needed < 0 else self.padding_needed # Safeguard against negative padding
        
        # Define the convolutional layer
        self.conv_down = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size, stride=downsample_divide)
        
    def forward(self, x):
        x = F.pad(x, (0, self.padding_needed))
        return self.conv_down(x)


class SinActivation(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)


class ConvBlock1D(nn.Module):
    """
    Double conv block, I leave the change in dimensions to the encoder and decoder classes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation_type: str='gelu'):
        super().__init__()
        assert activation_type in ['gelu', 'sin'], 'unknown activation type'
        if activation_type == 'gelu':
            activation_func = nn.GELU()
        elif activation_type == 'sin':
            activation_func = SinActivation()
        self.architecture = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            activation_func,
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            activation_func,
        )
           
            
    def forward(self, x):
        return self.architecture(x)
        


class Encoder1D(nn.Module):
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
            [ConvDownsample(kernel_size=5, downsample_divide=dim_change_param, in_dim=channel_list[idx + 1]) 
             for idx, dim_change_param in enumerate(dim_change_list)]
        )
        
        
    def forward(self, x):
        
        x = self.init_conv(x)
        
        for conv, dim_change in zip(self.conv_list, self.dim_change_list):
            x = conv(x)
            x = dim_change(x)
        
        return x


class Decoder1D(nn.Module):
    
    
    def __init__(self, channel_list: List[int], dim_change_list: List[int], input_channels: int=1, sin_locations: List[int]=None,
                 bottleneck_kernel_size: int=31):
        
        super().__init__()
        assert len(channel_list) == len(dim_change_list) + 1, "The channel list length must be greater than the dimension change list by 1"
        assert bottleneck_kernel_size % 2 == 1 or bottleneck_kernel_size == 0, \
            f"The bottleneck kernel size {bottleneck_kernel_size} must be an odd number or zero."
        
        self.bottleneck_kernel_size = bottleneck_kernel_size
        
        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(channel_list[-1], input_channels, kernel_size=3, padding=1)
        if bottleneck_kernel_size != 0:
            self.conv_1d_end = nn.Sequential(
                nn.Conv1d(input_channels, channel_list[1], kernel_size=bottleneck_kernel_size, padding=bottleneck_kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(channel_list[1], input_channels, kernel_size=bottleneck_kernel_size, padding=bottleneck_kernel_size // 2)
            )
        self.conv_list = nn.ModuleList(
            [ConvBlock1D(channel_list[idx], channel_list[idx + 1], 5, activation_type='gelu') 
             for idx in range(len(dim_change_list))]
        )
        self.dim_change_list = nn.ModuleList(
            [nn.ConvTranspose1d(channel_list[idx + 1], channel_list[idx + 1], 
                                kernel_size=dim_change_list[idx] + 12, stride=dim_change_list[idx], padding=6)
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
            
        return x_out + self.conv_1d_end(x_out) if self.bottleneck_kernel_size != 0 else x_out


class VQ1D(nn.Module):
    
    
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
        


class MultiLvlVQVariationalAutoEncoder(BaseNetwork):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """
    
    def __init__(self, 
                 sample_rate: int,
                 slice_length: float,
                 hidden_size: int,
                 latent_depth: int,
                 loss_dict: Dict[str, float],
                 vocabulary_size: int=8192,
                 bottleneck_kernel_size: int=31,
                 input_channels: int=1,
                 sin_locations: List[int] = None,
                 channel_dim_change_list: List[int] = [2, 2, 2, 4, 4],
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Parse arguments
        self.loss_dict = loss_dict
        self.cfg = kwargs
        self.sample_rate = sample_rate
        self.slice_length = slice_length
        self.samples_per_slice = int(sample_rate * slice_length)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.vocabulary_size = vocabulary_size
        self.channel_dim_change_list = channel_dim_change_list
        
        self.beta_factor = loss_dict['loss_beta']
        self.mel_factor = loss_dict['loss_mel']
        self.mel_factor_sub_1 = loss_dict['loss_mel_sub_1']
        self.mel_factor_sub_2 = loss_dict['loss_mel_sub_2']
        self.reconstruction_factor = loss_dict['loss_reconstruction']
        
        # Initialize mel spectrogram, TODO: Might do multiple ones for multiple losses
        self.mel_spec = None
        if 'mel_spec_config' in kwargs:
            self.mel_spec_config = kwargs['mel_spec_config']
            self.mel_spec = MelSpectrogram(sample_rate=sample_rate, **self.mel_spec_config)
            
        # Initialize the sub mel specs for additional losses
        self.mel_spec_sub_1 = None
        if 'mel_spec_sub_1_config' in kwargs:
            self.mel_spec_config_sub_1 = kwargs['mel_spec_sub_1_config']
            self.mel_spec_sub_1 = MelSpectrogram(sample_rate=sample_rate, **self.mel_spec_config_sub_1)
            
        # Initialize the sub mel specs for additional losses
        self.mel_spec_sub_2 = None
        if 'mel_spec_sub_2_config' in kwargs:
            self.mel_spec_config_sub_2 = kwargs['mel_spec_sub_2_config']
            self.mel_spec_sub_2 = MelSpectrogram(sample_rate=sample_rate, **self.mel_spec_config_sub_2)
        
        # Encoder parameter initialization
        encoder_channel_list = [hidden_size * (2 ** (idx + 1)) for idx in range(len(channel_dim_change_list))] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = list(reversed(encoder_channel_list))
        decoder_dim_changes = list(reversed(channel_dim_change_list))
        
        # Initialize network parts
        self.encoder = Encoder1D(encoder_channel_list, encoder_dim_changes, input_channels=input_channels)
        self.decoder = Decoder1D(decoder_channel_list, decoder_dim_changes, sin_locations=sin_locations, 
                                   bottleneck_kernel_size=bottleneck_kernel_size, input_channels=input_channels)
        self.vq_module = VQ1D(latent_depth, num_tokens=vocabulary_size)
        
        
    def forward(self, x: torch.Tensor, extract_losses: bool=False):
        
        origin_shape = x.shape
        x = x.reshape((x.shape[0], 1, -1))
        
        z_e = self.encoder(x)
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output['v_q'])
        
        total_output = {**vq_block_output,
                        'output': x_out.view(origin_shape)}
        
        if extract_losses:
            
            total_output.update({'reconstruction_loss': self._phased_loss(x, x_out, F.l1_loss)})
            
            if self.mel_spec is not None:
                total_output.update({'stft_loss': F.l1_loss(self._mel_spec_and_process(x), 
                                                            self._mel_spec_and_process(x_out))})
            if self.mel_spec_sub_1 is not None:
                total_output.update({'stft_loss_sub_1': F.l1_loss(self._mel_spec_sub_1_and_process(x), 
                                                                  self._mel_spec_sub_1_and_process(x_out))})
            if self.mel_spec_sub_2 is not None:
                total_output.update({'stft_loss_sub_2': F.l1_loss(self._mel_spec_sub_2_and_process(x), 
                                                                  self._mel_spec_sub_2_and_process(x_out))})
        
        return total_output
    
    
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
    
    
    def training_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = self.reconstruction_factor * total_output['reconstruction_loss'] +\
                     self.beta_factor * total_output['commitment_loss'] +\
                     total_output['alignment_loss'] +\
                     self.mel_factor * total_output['stft_loss'] +\
                     self.mel_factor_sub_1 * total_output['stft_loss_sub_1'] +\
                     self.mel_factor_sub_2 * total_output['stft_loss_sub_2']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Training {displayed_key}', value)
        self.log('Training total loss', total_loss)
        
        return total_loss
    
    
    def _phased_loss(self, x: torch.Tensor, x_target: torch.Tensor, loss_function, phase_parameter: int=10):
        
        loss_vector = torch.zeros(phase_parameter * 2).to(self.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = loss_function(x, x_target)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
            else:
                loss_vector[idx * 2] = loss_function(x[:, :, idx:], x_target[:, :, :-idx])
                loss_vector[idx * 2 + 1] = loss_function(x[:, :, :-idx], x_target[:, :, idx:])
        return loss_vector.min()
    
    
    def validation_step(self, batch, batch_idx):
        
        music_slice = batch['music slice']
        total_output = self.forward(music_slice, extract_losses=True)
        total_loss = self.reconstruction_factor * total_output['reconstruction_loss'] +\
                     self.beta_factor * total_output['commitment_loss'] +\
                     total_output['alignment_loss'] +\
                     self.mel_factor * total_output['stft_loss'] +\
                     self.mel_factor_sub_1 * total_output['stft_loss_sub_1'] +\
                     self.mel_factor_sub_2 * total_output['stft_loss_sub_2']
            
        for key, value in total_output.items():
            if 'loss' in key.split('_'):
                displayed_key = key.replace('_', ' ')
                self.log(f'Validation {displayed_key}', value)
        self.log('Validation total loss', total_loss, prog_bar=True)