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
        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg['base_loss_type']]
        
    def forward(self, x, x_target):
        
        emb = x['emb']
        z_e = x_target['z_e']
        
        return self.base_loss_type(emb, z_e.detach())
    
    
class CommitLoss(LossBase):
    '''
    VQ-VAE codebook commitment loss
    '''
    
    def __init__(self, loss_cfg: Dict):
        
        super().__init__(loss_cfg)
        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg['base_loss_type']]
        
    def forward(self, x, x_target):
        
        emb = x['emb']
        z_e = x_target['z_e']

        return self.base_loss_type(emb.detach(), z_e)
        