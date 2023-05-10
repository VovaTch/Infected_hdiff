from typing import OrderedDict, List, Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm

from models.base import BaseNetwork
from models.diffusion_vit import DiffusionViT
from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from .base import Integrator
from utils.other import load_cfg_dict


class InfectedHDiffIntegrator(Integrator):
    """
    This class integrates a cascading series of diffusion models and vqvae's for music generation.
    """
    
    def __init__(self, 
                 modules: OrderedDict[str, str], 
                 weight_paths: Dict[str, str],
                 batch_size: int = 4, 
                 device: str = 'cuda', 
                 batch_length: int=4096, 
                 reduction_list: List[int]=[8, 8, 8, 2], 
                 latent_sizes: List[int]=[8, 8, 8, 8],
                 **kwargs):
        
        super().__init__(modules, batch_size, device, **kwargs)
        self.reduction_list = reduction_list
        self.batch_length = batch_length
        self.latent_sizes = latent_sizes
        self.weight_paths = weight_paths
        
        self.track_length = None


    @torch.no_grad()
    def __call__(self, track_length: float = 300):
        
        self.track_length = track_length
        return self
                    
                    
    def __iter__(self):
        """
        This iterator will be used to eventually generate data for the React demo app.
        """
        
        assert self.track_length is not None, "Must input track length via the '__call__' method."
        
        # Create the initial noise
        reduction_factor = np.prod(self.reduction_list)
        number_of_sample_vectors = int(self.track_length * 44100 / (reduction_factor * self.latent_sizes[-1] * 8)) # This multiplication by 8 is a hack
        init_noise = torch.randn((1, self.latent_sizes[-1], number_of_sample_vectors)).to(self.device)
        running_data = init_noise
        
        module_idx = 0
        
        for module_name, module_cfg in self.module_cfgs.items():
            
            cfg = load_cfg_dict(module_cfg)
            if 'diff' in module_name.split('_'):
                module = DiffusionViT.load_from_checkpoint(self.weight_paths[module_name], **cfg).to(self.device)
            elif 'vqvae' in module_name.split('_'):
                module = MultiLvlVQVariationalAutoEncoder.load_from_checkpoint(self.weight_paths[module_name], **cfg).to(self.device)
            else:
                raise Exception(f'Unknown module for name {module_name}')
            
            dataloader = self._create_dataloader(running_data)
            result_collector = None
            
            # Initial yield to produce the additional progress bar
            running_data_idx = 0
            yield [module_name, module_idx, running_data_idx / len(dataloader) * 100] 
            # Module name, module index, percent completed
            
            for data in tqdm.tqdm(dataloader, f'Running network {module_name}'):
                
                data = data[0].to(self.device)
                if isinstance(module, DiffusionViT):
                    output = module.denoise(data) # TODO: Change it such that it can output running data into an outside script
                else:
                    output = module.decode(data)
                    
                if result_collector is None:
                    result_collector = output
                else:
                    result_collector = torch.cat((result_collector, output), dim=0)
                    
                running_data_idx += 1
                yield [module_name, module_idx, running_data_idx / len(dataloader) * 100] 
                    
            print(f'Network {module_name} has finished inference.')
            del module # Release the module from memory and make way for the next one
            running_data = result_collector
            module_idx += 1
            
        self.track = result_collector.flatten()   
        return self.track      
                    
        
    def _create_dataloader(self, data: torch.Tensor):
        """
        This assumes 1 channel music data with a structure of BS x Latent x Length
        """
        
        data_flattened = data.reshape((data.shape[1], 1, -1)).permute((1, 0, 2))
        padding_size = data.shape[2] % self.batch_length if data_flattened.shape[2] > self.batch_length else 0
        data_padded = F.pad(data_flattened, (0, self.batch_length - padding_size if padding_size != 0 else 0))
        dataset = TensorDataset(data_padded.permute((1, 0, 2)).reshape(-1, data.shape[1], 
                                                                       self.batch_length if data_flattened.shape[2] >\
                                                                           self.batch_length else data.shape[2]))
        return DataLoader(dataset, self.batch_size)
        
