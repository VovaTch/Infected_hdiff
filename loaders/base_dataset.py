import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import tqdm

from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder


class BaseLatentLoader(Dataset):
    
    def __init__(self, 
                 collection_parameter: int,
                 slice_length: int,
                 preload_file_path: str,
                 preload_metadata_file_path: str,
                 device: str="cpu",
                 prev_dataset=None,
                 prev_vqvae: MultiLvlVQVariationalAutoEncoder=None,
                 preload: bool=True,
                 generative: bool=False,
                 **kwargs):
        
        # Initialize the object variables
        super().__init__()
        
        self.prev_dataset = prev_dataset
        self.prev_vqvae = prev_vqvae
        self.device = device
        self.preload = preload
        self.preload_file_path = preload_file_path
        self.collection_parameter = collection_parameter
        self.slice_length = slice_length
        self.metadata = None
        self.generative = generative
        
        
        # Preload the data
        if self.preload:
            
            # Load pt file if exists
            if os.path.isfile(preload_file_path):
                print(f'Loading file {preload_file_path}...')
                self.processed_slice_data = torch.load(preload_file_path)
                print(f'Music file {preload_file_path} is loaded.')
                
            # Save pt file if it doesn't
            else:
                assert prev_vqvae is not None and prev_dataset is not None, 'If no dataset file exists, must have vqvae and dataset.'
                self.processed_slice_data, self.metadata = self._create_latents()
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
                    _, self.metadata = self._create_latents()
                with open(preload_metadata_file_path, 'wb') as f:
                    pickle.dump(self.metadata, f)    
                print(f'Saved music file at {preload_metadata_file_path}')
                
        else:
            assert prev_vqvae is not None and prev_dataset is not None, 'When not preloading the lvl1 vqvae and the dataset must be set.'
            
        del self.prev_vqvae
        
        self.track_list_unique = []
        for track_name in self.metadata:
            if track_name[0] not in self.track_list_unique:
                self.track_list_unique.append(track_name[0])
        
    @torch.no_grad()
    def _create_latents(self):
        """
        Runs the level 1 vqvae network to create the latents and save them.
        """
        
        width, length = self._compute_latent_dims()
        data_collector = torch.zeros((0, width, length * self.collection_parameter)).to(self.device)
        loader = DataLoader(self.prev_dataset, batch_size=1)
        
        # Initialize running variables
        prev_track_name = None
        latent_collector = None
        running_idx = 0
        track_name_list = []
        
        for batch in tqdm.tqdm(loader, 'Loading music slices...'):
            music_slice, current_track_name = batch['music slice'].to(self.device), batch['track name'][0]
            latent = self.prev_vqvae.encode(music_slice)
            latent = self.prev_vqvae.vq_module(latent)['v_q']
            
            # If the collector is filled, reset the collector
            if running_idx % self.collection_parameter == 0:
                if latent_collector is not None:
                    data_collector = torch.cat((data_collector, latent_collector.unsqueeze(0)), dim=0)
                    track_name_list.append([current_track_name])
                latent_collector = torch.zeros((width, 0)).to(self.device)
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
            
        # Collect the remainder after the loop
        if latent_collector.shape[1] > 0:
            padding = length * self.collection_parameter - latent_collector.shape[1]
            latent_collector = F.pad(latent_collector, (0, padding))
            data_collector = torch.cat((data_collector, latent_collector.unsqueeze(0)), dim=0)
            track_name_list.append([current_track_name]) 
                
        return data_collector, track_name_list
        
        
    def _compute_latent_dims(self):
        
        length = self.slice_length
        for dim_change in self.prev_vqvae.channel_dim_change_list:
            length /= dim_change
            print(length)
        width = self.prev_vqvae.latent_depth
        
        return int(width), int(length)
    
    
    def __len__(self):
        return self.processed_slice_data.shape[0] if self.preload else 100
    
    
    def __getitem__(self, idx):
        
        if self.preload:
            slice = self.processed_slice_data[idx]
            track_name = self.metadata[idx]
        else:
            raise NotImplemented('Currently this dataset only functions with preloading')
            
        batch = {'music slice': slice.to(self.device), 'track name': track_name[0], 
                 'track index': torch.tensor(self.track_list_unique.index(track_name[0])).int()}
        if self.generative:
            
            if idx == 0:
                back_cond_slice = torch.zeros_like(slice.to(self.device))
            else:
                back_cond_slice = self.processed_slice_data[idx - 1].to(self.device)
                
            if idx == self.__len__() - 1:
                forward_cond_slice = torch.zeros_like(slice.to(self.device))
            else:
                forward_cond_slice = self.processed_slice_data[idx + 1].to(self.device)
            
            batch.update({'back conditional slice': back_cond_slice, 'forward conditional slice': forward_cond_slice})
            
        return batch