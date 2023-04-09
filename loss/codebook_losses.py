from typing import Dict

import torch
from torchaudio.transforms import MelSpectrogram

from .base import LossBase, BASE_LOSS_TYPES


class AlignLoss(LossBase):
    '''
    VQ-VAE codebook alignment loss
    '''
    
    def __init__(self, loss_cfg: Dict):
        
        super().__init__(loss_cfg)
        
    def forward(self, x, x_target):
        
        emb = x_target['emb']
        z_e = x['z_e']
        
        return torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
    
    
class CommitLoss(LossBase):
    '''
    VQ-VAE codebook commitment loss
    '''
    
    def __init__(self, loss_cfg: Dict):
        
        super().__init__(loss_cfg)
        
    def forward(self, x, x_target):
        
        emb = x_target['emb']
        z_e = x['z_e']
        
        return torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))
        