from typing import OrderedDict, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import tqdm

from models.base import BaseNetwork
from models.diffusion_vit import DiffusionViT
from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from .base import Integrator
from utils.other import load_cfg_dict


LATENT_LENGTH = 512

class InfectedHDiffIntegrator(Integrator):
    """
    This class integrates a cascading series of diffusion models and vqvae's for music generation.
    """
    
    def __init__(self, 
                 modules: OrderedDict[str, str], 
                 weight_paths: Dict[str, str],
                 batch_size: int = 4, 
                 batch_divide: int = 8,
                 device: str = 'cuda', 
                 batch_length: int = 32768, 
                 sample_rate: int = 44100,
                 **kwargs):
        
        super().__init__(modules, batch_size, device, **kwargs)
        self.reduction_list = []
        self.batch_length = batch_length
        self.latent_sizes = []
        self.weight_paths = weight_paths
        self.sample_rate = sample_rate
        self.batch_divide = batch_divide
        
        self.track_length = None


    @torch.no_grad()
    def __call__(self, track_length: float = 300, conditionals: Dict[int, int]=None):
        self.track_length = track_length
        self.conditionals = conditionals
        return self
    
    
    def __iter__(self):
        """
        This iterator will be used to eventually generate data for the React demo app.
        """
        
        assert self.track_length is not None, "Must input track length via the '__call__' method."
        
        # Load modules
        modules = self._load_modules()
        
        # Create the initial noise
        reduction_factor = np.prod(self.reduction_list)
        self.number_of_sample_vectors = int(self.track_length * self.sample_rate / reduction_factor) # TODO: Check if works, for 6 minutes should be around ~500
        init_noise = torch.randn((1, self.latent_sizes[-1], self.number_of_sample_vectors)).to(self.device)
        init_noise = self._right_pad_if_necessary(init_noise, LATENT_LENGTH) # TODO: Change LATENT_LENGTH to something not hardcoded?
        
        init_noise = init_noise.permute((0, 2, 1)).reshape((-1, 512, 8)).permute((0, 2, 1)) # 2 (3) x 8 x 512
        
        # Run the models
        self.track = self._run_modules(modules, init_noise)
        
        
    def _load_modules(self) -> List[nn.Module]:
        
        # Create a module dict, and record all the reduction sizes for later
        modules = []
        self.latent_sizes = []
        self.reduction_list = []
        
        for module_name, module_cfg in self.module_cfgs.items():
        
            cfg = load_cfg_dict(module_cfg)
            if 'diff' in module_name.split('_'):
                module = DiffusionViT.load_from_checkpoint(self.weight_paths[module_name], **cfg).to(self.device)
            elif 'vqvae' in module_name.split('_'):
                module = MultiLvlVQVariationalAutoEncoder.load_from_checkpoint(self.weight_paths[module_name], **cfg).to(self.device)
                self.latent_sizes.append(cfg['latent_depth'])
                self.reduction_list.append(np.prod(cfg['channel_dim_change_list']) * cfg['input_dim'] // cfg['latent_depth'])
            else:
                raise Exception(f'Unknown module for name {module_name}')
            
            modules.append(module)
            
        return modules
    
    
    @torch.no_grad()
    def _run_modules(self, modules: List[Tuple[str, nn.Module]], data: torch.Tensor) -> torch.Tensor:
        """
        Runs the first model in the list. The data is assumed, at the current state, size BS x 8 x 512
        """
        
        current_model = modules[0]
        bs = data.shape[0]
        
        # Breaking condition
        if len(modules) == 0:
            return data
        
        if 'diff' in current_model[0]:
            output = current_model[1].denoise(data)
            recursive_output = self._run_modules(modules[1:], output) # BS x 8 x 512
            
        elif 'vqvae' in current_model[0]:
            output_ind, _ = current_model[1].decode(data) # BS x 8 x 4096 or BS x 1 x 32768
            cfg = load_cfg_dict(self.module_cfgs[current_model[0]])
            output = output_ind.permute((0, 2, 1)).reshape((cfg['input_channels'] * bs, -1, cfg['latent_depth'])).permute((0, 2, 1)) # BS * 8 x 8 x 512
            
            output_dataset = TensorDataset(output)
            output_dataloader = DataLoader(output_dataset, batch_size=1)
            
            # Run the dataset recursively
            recursive_output = torch.zeros((0, cfg['input_channels'], ))
            for batch in tqdm.tqdm(output_dataloader, f"Running model {current_model[0]}..."):
                recursive_output_ind = self._run_modules(modules[1:], batch[0]) # batch[0] : 1 x 8 x 512
                recursive_output_ind = recursive_output_ind.permute((0, 2, 1)).\
                    reshape((cfg['input_channels'] * bs, -1, cfg['latent_depth'])).permute((0, 2, 1)) # BS * 8 x 8 x 512
                recursive_output = torch.cat((recursive_output, recursive_output_ind), dim=0)
                
        return recursive_output
    
    
    @staticmethod()
    def _right_pad_if_necessary(data: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Assumes the data is in the shape of 1 x 8 x N, transforms it to 1 x 8 x 512
        """
        if data.shape[2] < max_length:
            data = F.pad(data, (0, max_length - data.shape[2]))
        return data
            
            
        
        
            
        
            
    
    