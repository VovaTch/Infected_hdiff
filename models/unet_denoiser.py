import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl

import loaders


DATASETS = {'denoising_dataset': loaders.DenoiseDataset,
            'music_slice_dataset': loaders.MP3SliceDataset}


class WaveUNet_Denoiser(pl.LightningModule):
    
    def __init__(self, 
                 learning_rate: float,
                 weight_decay: float,
                 batch_size: int,
                 epochs: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 filter_size_encoder: int, 
                 filter_size_decoder: int,
                 num_input_channels: int=1,
                 num_filters: int=1,
                 dataset_name: str='denoising_dataset',
                 dataset_path: str='data/music_samples',
                 scheduler_type: str='one_cycle_lr',
                 eval_split_factor: float=0.01,
                 **kwargs):
        
        super().__init__()
        
        # Initialize variables
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.filter_size_encoder = filter_size_encoder
        self.filter_size_decoder = filter_size_decoder
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.cfg = kwargs
        self.dataset_path = dataset_path
        
        # Initialize channel lists
        enc_channel_in = [self.num_input_channels] + [min(self.num_decoder_layers, (i + 1)) * self.num_filters 
                                                      for i in range(self.num_encoder_layers - 1)]
        enc_channel_out = [min(self.num_decoder_layers, (i + 1)) * self.num_filters for i in range(self.num_encoder_layers)]
        dec_channel_out = enc_channel_out[:self.num_decoder_layers][::-1]
        dec_channel_in = [enc_channel_out[-1]*2 + self.num_filters] + [enc_channel_out[-i-1] + dec_channel_out[i-1] 
                                                                       for i in range(1, self.num_decoder_layers)]
        
        # Initialize encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for i in range(self.num_encoder_layers):
            self.encoder.append(nn.Conv1d(enc_channel_in[i], enc_channel_out[i], self.filter_size_encoder))

        for i in range(self.num_decoder_layers):
            self.decoder.append(nn.Conv1d(dec_channel_in[i], dec_channel_out[i], self.filter_size_decoder))

        self.middle_layer = nn.Sequential(
            nn.Conv1d(enc_channel_out[-1], enc_channel_out[-1] + self.num_filters, self.filter_size_encoder),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Conv1d(self.num_filters + self.num_input_channels, self.num_input_channels, kernel_size=1),
            nn.Tanh()
        )
        
        assert dataset_name in DATASETS, f'Dataset {dataset_name} is not in the datasets options.'
        assert 0 <= eval_split_factor <= 1, f'The split factor must be between 0 and 1, current value is {eval_split_factor}'
        self.dataset = None
        self.eval_split_factor = eval_split_factor
        self.dataset_name = dataset_name
        
        # Optimizers
        assert scheduler_type in ['none', 'one_cycle_lr', 'reduce_on_platou'] # TODO fix typo, program the schedulers in
        self.scheduler_type = scheduler_type
        
        
    def forward(self, x: torch.Tensor):
        
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
            x = F.interpolate(x, size=x.shape[-1] * 2 - 1, mode='linear', align_corners=True)
            x = self._crop_and_concat(x, encoder[self.num_encoder_layers - i - 1])
            x = self.decoder[i](x)
            x = F.leaky_relu(x, 0.2)

        # Concat with original input
        x = self._crop_and_concat(x, input)

        # Output prediction
        output_vocal = self.output_layer(x)
        output_accompaniment = self._crop(input, output_vocal.shape[-1]) - output_vocal
        return output_vocal, output_accompaniment
    
    
    def _crop_and_concat(self, x1: torch.Tensor, x2: torch.Tensor):
        crop_x2 = self._crop(x2, x1.shape[-1])
        x = torch.cat([x1,crop_x2], dim=1)
        return x

    def _crop(self, tensor: torch.Tensor, target_shape):
        # Center crop
        shape = tensor.shape[-1]
        diff = shape - target_shape
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:,:,crop_start:-crop_end]
    
    
    def configure_optimizers(self):
        self._set_dataset()
        if len(self.dataset) % self.batch_size == 0:
            total_steps = len(self.dataset) // self.batch_size
        else:
            total_steps = len(self.dataset) // self.batch_size + 1
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)#, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'one_cycle_lr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                                            epochs=self.epochs,
                                                            steps_per_epoch=total_steps)
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1e6)
            
        elif self.scheduler_type == 'none':
            return [optimizer]
            
        scheduler_settings= {'scheduler': scheduler,
                             'interval': 'step',
                             'monitor': 'Training reconstruction loss',
                             'frequency': 1} # Set the scheduler to step every batch
        
        return [optimizer], [scheduler_settings]
    
    
    def _set_dataset(self):
        
        if self.dataset is None:
            self.dataset = DATASETS[self.dataset_name](**self.cfg, audio_dir=self.dataset_path)
            train_dataset_length = int(len(self.dataset) * (1 - self.eval_split_factor))
            self.train_dataset, self.eval_dataset = random_split(self.dataset, 
                                                                (train_dataset_length, 
                                                                len(self.dataset) - train_dataset_length))