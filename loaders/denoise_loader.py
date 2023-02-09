import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import csv
import numpy as np

if TYPE_CHECKING:
    from .music_loader import MP3SliceDataset
    from models.level_1_vqvae import Lvl1VQVariationalAutoEncoder


class DenoiseDataset(Dataset):
    
    def __init__(self,
                 vqvae: 'Lvl1VQVariationalAutoEncoder',
                 music_loader: 'MP3SliceDataset',
                 loader_batch_size: int=8,
                 preload_file_path: str='data/lvl1_outs_tensor.pt',
                 preload: bool=True):
        
        # Make sure that all the data is in a single file or is loaded
        self.music_loader = music_loader
        assert self.music_loader.preload, 'The loader must have preload activated'
        
        # Initiate variables
        super().__init__()
        self.vqvae = vqvae
        self.loader_batch_size = loader_batch_size
        self.post_vqvae_data = None
        self.preload = preload
        
        if self.preload:
            
            # Load pickle file if exists
            if os.path.isfile(preload_file_path):
                print(f'Loading file {preload_file_path}...')
                self.post_vqvae_data = torch.load(preload_file_path)
                print(f'Music file {preload_file_path} is loaded.')
                
            # Save pickle file if not
            else:
                self._create_dataset()
                torch.save(self.post_vqvae_data, preload_file_path)
                print(f'Saved music file at {preload_file_path}')
        
        
    def _create_dataset(self):
        '''
        Run the network over all the data
        '''
        
        loader = DataLoader(self.music_loader, batch_size=self.loader_batch_size, shuffle=False)
        self.post_vqvae_data = torch.zeros_like(self.music_loader.processed_slice_data).to(self.vqvae.device)
        with torch.no_grad():
            for idx, batch in tqdm.tqdm(enumerate(loader), 'Processing raw data...'):
                batch_img = batch['music slice'].to(self.vqvae.device)
                vqvae_outs = self.vqvae(batch_img)
                self.post_vqvae_data[idx * self.loader_batch_size: (idx + 1) * 
                                     self.loader_batch_size, :] = vqvae_outs['output'].squeeze(1)
                if idx >= 50:
                    break
        # self.post_vqvae_data_np = self.post_vqvae_data.numpy()
            
    
    def __len__(self):
        return self.post_vqvae_data.shape[0] if self.preload else 100
    
    
    def __getitem__(self, idx):
        
        batch = {'music slice', self.music_loader.processed_slice_data[idx]}
        
        if self.preload:
            batch['vqvae outs'] = self.post_vqvae_data[idx]
        else:
            batch['vqvae outs'] = self.vqvae(self.music_loader.processed_slice_data[idx])
            
        return batch
            
        
        