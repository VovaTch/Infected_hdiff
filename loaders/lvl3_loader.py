from typing import TYPE_CHECKING
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle

if TYPE_CHECKING:
    from .lvl2_loader import Lvl3InputDataset
    from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder


SLICE_LENGTH = 32768 # TODO: make it not hardcoded


class Lvl3InputDataset(Dataset):
    """
    This class takes the encoded level 2 vqvae latent vectors and collects them all in a .pt file, to be loaded for the level 2 training.
    The level 3 vqvae training will take 8 latents concatenated together as a sample.
    """
    
    def __init__(self,
                 collection_parameter: int=8,
                 lvl2_dataset: 'Lvl3InputDataset'=None,
                 lvl2_vqvae: 'MultiLvlVQVariationalAutoEncoder'=None,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/002-datatensor.pt",
                 preload_metadata_file_path: str='data/music_samples/002-metadata.pkl',
                 **kwargs):
        
        # Initialize the object variables
        super().__init__()
        self.lvl2_dataset = lvl2_dataset
        self.lvl2_vqvae = lvl2_vqvae
        self.device = device
        self.preload = preload
        self.preload_file_path = preload_file_path
        self.collection_parameter = collection_parameter
        
        # Preload the data
        if self.preload:
            
            # Load pt file if exists
            if os.path.isfile(preload_file_path):
                print(f'Loading file {preload_file_path}...')
                self.processed_slice_data = torch.load(preload_file_path)
                print(f'Music file {preload_file_path} is loaded.')
                
            # Save pt file if it doesn't
            else:
                assert lvl2_vqvae is not None and lvl2_dataset is not None, 'If no dataset file exists, must have vqvae and dataset.'
                self.processed_slice_data, self.metadata = self._create_lvl2_latents()
                self.processed_slice_data = self.processed_slice_data.to(device)
                torch.save(self.processed_slice_data, preload_file_path)
                print(f'Saved music file at {preload_file_path}')
                
            # Load pickle file if exists
            if os.path.isfile(preload_metadata_file_path):
                print(f'Loading file {preload_metadata_file_path}...')
                with open(preload_metadata_file_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f'Music file {preload_metadata_file_path} is loaded.')
                
            # Save pickle file if not
            else:
                if self.metadata is None:
                    _, self.metadata = self._create_lvl2_latents()
                with open(preload_metadata_file_path, 'wb') as f:
                    pickle.dump(self.metadata, f)    
                print(f'Saved music file at {preload_metadata_file_path}')
                
        else:
            assert lvl2_vqvae is not None and lvl2_dataset is not None, 'When not preloading the lvl2 vqvae and the dataset must be set.'
            
        del self.lvl2_vqvae
                
    
    @torch.no_grad()
    def _create_lvl2_latents(self):
        """
        Runs the level 1 vqvae network to create the latents and save them.
        """
        
        width, length = self._compute_latent_dims()
        data_collector = torch.zeros((0, width, length * self.collection_parameter)).to(self.device)
        loader = DataLoader(self.lvl2_dataset, batch_size=1)
        
        # Initialize running variables
        prev_track_name = None
        latent_collector = None
        running_idx = 0
        track_name_list = []
        
        for batch in tqdm.tqdm(loader, 'Loading music slices...'):
            music_slice, current_track_name = batch['music slice'].to(self.device), batch['track name'][0]
            music_slice = music_slice.reshape((music_slice.shape[0], self.lvl2_vqvae.input_channels, -1))
            latent = self.lvl2_vqvae.encoder(music_slice)
            latent = self.lvl2_vqvae.vq_module(latent)['v_q']
            
            # If the collector is filled, reset the collector
            if running_idx % self.collection_parameter == 0:
                if latent_collector is not None:
                    data_collector = torch.cat((data_collector, latent_collector.unsqueeze(0)), dim=0)
                    track_name_list.append([current_track_name])
                latent_collector = torch.zeros(width, 0).to(self.device)
                running_idx = 0
                
            latent_collector = torch.cat((latent_collector, latent.squeeze(0)), dim=1)
            running_idx += 1
            
            # If the track switches, pad and reset the latent collector
            if prev_track_name is not None and current_track_name != prev_track_name:
                padding = length * self.collection_parameter - latent_collector.shape[1]
                latent_collector = F.pad(latent_collector, (0, padding))
                data_collector = torch.cat((data_collector, latent_collector.unsqueeze(0)), dim=0)
                track_name_list.append([current_track_name])
                latent_collector = None
                running_idx = 0
                
            prev_track_name = current_track_name
                
        return data_collector, track_name_list
        
        
    def _compute_latent_dims(self):
        
        length = SLICE_LENGTH
        for dim_change in self.lvl2_vqvae.channel_dim_change_list:
            length /= dim_change
        width = self.lvl2_vqvae.latent_depth
        
        return int(width), int(length)
    
    
    def __len__(self):
        return self.processed_slice_data.shape[0] if self.preload else 100
    
    
    def __getitem__(self, idx):
        
        if self.preload:
            slice = self.processed_slice_data[idx]
            track_name = self.metadata[idx]
        else:
            raise NotImplemented('Currently this dataset only functions with preloading')
            
        return {'music slice': slice.to(self.device), 'track name': track_name[0]}