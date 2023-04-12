from abc import ABC, abstractmethod
from typing import OrderedDict

import torch
import torchaudio

from models.base import BaseNetwork


class Integrator(ABC):
    
    
    def __init__(self, module_cfgs: OrderedDict[str, str], batch_size: int=4, device: str='cuda', **kwargs):
        
        super().__init__()
        
        assert len(module_cfgs) != 0, 'The module ordered dictionary cannot be empty!'
        
        self.module_cfgs = module_cfgs
        self.batch_size = batch_size
        self.device = device
        
        # This assumes the ordered dict is from the 1st to last model
        self.reset_integrator()
        self.track: torch.Tensor = None
        
        
    def reset_integrator(self):
        self.current_model_idx = 0
        self.current_model_cfg = list(self.module_cfgs.values())[self.current_model_idx]
        
        
    def _forward_model(self):
        self.current_model_idx += 1
        self.current_model_cfg = list(self.module_cfgs.values())[self.current_model_idx]
        
        
    @abstractmethod
    def __call__(self, track_length: float=300):
        raise NotImplementedError('Must implement a __call__ function for creating the integrator.')
    
    
    def save_track(self, path: str='src/output/sound/sample.mp3'):
        assert self.track is not None, 'Cannot save an empty track!'
        torchaudio.save(path, self.track.unsqueeze(0).cpu().detach(), 44100, format='mp3')
        print(f'Saved track at {path}')
        
        
        

        

        