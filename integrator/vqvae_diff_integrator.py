from typing import OrderedDict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm

from models.base import BaseNetwork
from models.diffusion_vit import DiffusionViT
from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from .base import Integrator


class InfectedHDiffIntegrator(Integrator):
    """
    This class integrates a cascading series of diffusion models and vqvae's for music generation.
    """
    
    def __init__(self, 
                 modules: OrderedDict[str, BaseNetwork], 
                 batch_size: int = 4, 
                 device: str = 'cuda', 
                 batch_length: int=4096, 
                 reduction_list: List[int]=[8, 8, 8, 2], 
                 latent_sizes: List[int]=[8, 8, 8, 8]):
        
        super().__init__(modules, batch_size, device)
        self.reduction_list = reduction_list
        self.batch_length = batch_length
        self.latent_sizes = latent_sizes


    def __call__(self, track_length: float = 300):
        
        # Create the initial noise
        reduction_factor = np.prod(self.reduction_list)
        number_of_sample_vectors = int(track_length * 44100 / (reduction_factor * self.latent_sizes[-1]))
        init_noise = torch.randn((1, self.latent_sizes[-1], number_of_sample_vectors)).to(self.device)
        running_data = init_noise
        
        for module_name, module_cfg in self.module_cfgs.items():
            
            if 'diff' in module_name:
                module = DiffusionViT(**module_cfg)
            elif 'vqvae' in module_name:
                module = MultiLvlVQVariationalAutoEncoder(**module_cfg)
            else:
                raise Exception(f'Unknown module for name {module_name}')
            
            dataloader = self._create_dataloader(running_data)
            result_collector = None
            
            for data in tqdm.tqdm(dataloader, f'Running network {module_name}'):
                
                data = data.to(self.device)
                if isinstance(module, MultiLvlVQVariationalAutoEncoder):
                    output = module.denoise(data) # TODO: Change it such that it can output running data into an outside script
                else:
                    output = module(data)['output']
                    
                if result_collector is None:
                    result_collector = output
                else:
                    result_collector = torch.cat((result_collector, output), dim=2)
                    
            print(f'Network {module_name} has finished inference.')
            del module # Release the module from memory and make way for the next one
            running_data = result_collector
            
        self.track = result_collector.flatten()         
                    
        
    def _create_dataloader(self, data: torch.Tensor):
        """
        This assumes 1 channel music data with a structure of BS x Latent x Length
        """
        
        data_flattened = data.reshape((data.shape[1], 1, -1)).permute((1, 0, 2))
        padding_size = data.shape[2] % self.batch_length if data_flattened.shape[2] > self.batch_length else 0
        data_padded = F.pad(data_flattened, (0, self.batch_length - padding_size if padding_size != 0 else 0))
        dataset = TensorDataset(data_padded.permute((1, 0, 2)).reshape(-1, data.shape[1], self.batch_length))
        return DataLoader(dataset, self.batch_size)
        
