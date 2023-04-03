from abc import abstractmethod
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F


BASE_LOSS_TYPES= {'l1_loss': F.l1_loss,
                  'l2_loss': F.mse_loss}

class LossBase(nn.Module):
    '''
    Abstract method for the type of loss
    '''
    
    def __init__(self, loss_cfg: Dict):
        
        super().__init__()
        self.loss_cfg = loss_cfg
        
        assert 'weight' in loss_cfg, 'The loss weight must be in the configuration dictionary.'
        self.weight = loss_cfg['weight']
    
    
    def get_weight(self):
        return self.weight
    
    
    @abstractmethod
    def forward(self, x, x_target):
        raise NotImplementedError('A forward function for the error must be implemented')
    
    