from torchaudio.transforms import MelSpectrogram
import pytorch_lightning as pl
import torch.nn as nn

from .base import BaseNetwork


class Block2D(nn.Module):
    """
    Double conv block,
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.GELU(),
        )
           
            
    def forward(self, x):
        return self.architecture(x)


class InfectedReconstructor(BaseNetwork):
    
    def __init__(self, 
                 hidden_size: int,
                 **kwargs) -> pl.LightningModule:
        super().__init__(**kwargs)
        
        # Initialize variables
        self.hidden_size = hidden_size
        
        # Initialize mel spectrogram, TODO: Might do multiple ones for multiple losses
        self.mel_spec = None
        assert 'mel_spec_config' in kwargs, 'There must be a configuration dict in the configuration file for the mel spectrogram'
        self.mel_spec_config = kwargs['mel_spec_config']
        self.mel_spec = MelSpectrogram(sample_rate=kwargs['sample_rate'], **self.mel_spec_config)
            
        # TODO: check if this is correct
        self.time_dim = self.mel_spec_config['n_fft'] // 2 + 1
        self.mel_dim = self.mel_spec_config['n_mels']
        self.signal_length = (self.time_dim - 1) * self.mel_spec_config['hop_length']
        
        # Architecture
        self.init_conv = nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=1)
        self.inner_blocks = nn.ModuleList([])
        
            
        