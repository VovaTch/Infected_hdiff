from typing import TYPE_CHECKING
import os
from typing import List
import tempfile
import random

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import tqdm

if TYPE_CHECKING:
    from .music_loader import MP3SliceDataset
    from models.level_1_vqvae import Lvl1VQVariationalAutoEncoder


SLICE_LENGTH = 32768 # TODO: make it not hardcoded


class Lvl2InputDataset(Dataset):
    """
    This class takes the encoded level 1 vqvae latent vectors and collects them all in a .pt file, to be loaded for the level 2 training.
    The level 2 vqvae training will take 8 latents concatenated together as a sample.
    """
    
    def __init__(self,
                 collection_parameter: int=8,
                 lvl1_dataset: 'MP3SliceDataset'=None,
                 lvl1_vqvae: 'Lvl1VQVariationalAutoEncoder'=None,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/001-datatensor.bt",
                 **kwargs):
        
        # Initialize the object variables
        super().__init__()
        self.lvl1_dataset = lvl1_dataset.processed_slice_data
        self.lvl1_vqvae = lvl1_vqvae
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
                assert lvl1_vqvae is not None and lvl1_dataset is not None, 'If no dataset file exists, must have vqvae and dataset.'
                self.processed_slice_data = self._create_lvl1_latents()
                torch.save(self.processed_slice_data, preload_file_path)
                print(f'Saved music file at {preload_file_path}')
                
                
    def _create_lvl1_latents(self):
        """
        Runs the level 1 vqvae network to create the latents and save them.
        """
        
        width, length = self._compute_latent_dims()
        data_collector = torch.zeros((0, self.collection_parameter, width, length)).to(self.device)
        loader = DataLoader(self.lvl1_dataset, batch_size=1)
        prev_track_name = None
        for idx, batch in tqdm(tqdm(enumerate(loader)), 'Loading music slices...'):
            music_slice, current_track_name = batch['music slice'].to(self.device), 
            latent = self.lvl1_vqvae(music_slice)
            
            
            
            data_collector = torch.cat((data_collector, latent), dim=0)
            
        
        
    def _compute_latent_dims(self):
        
        length = SLICE_LENGTH
        for dim_change in self.lvl1_vqvae.channel_dim_change_list:
            length /= dim_change
        width = self.lvl1_vqvae.latent_depth
        
        return width, length
            