from typing import Dict

import torch
import torch.nn as nn

from .mel_spec_loss import MelSpecLoss
from .rec_loss import RecLoss
from .codebook_losses import AlignLoss, CommitLoss


LOSS_TYPES = {'melspec': MelSpecLoss,
              'reconstruction': RecLoss,
              'alignment': AlignLoss,
              'commitment': CommitLoss}

class TotalLoss(nn.Module):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        # Initialize losses
        self.cfg = cfg
        self.loss_dict = {}
        self.loss_weight_dict = {}
        for loss_type in cfg:
            self.loss_dict[loss_type] = LOSS_TYPES[loss_type.split('_')[0]](cfg[loss_type])
            self.loss_weight_dict[loss_type] = self.loss_dict[loss_type].weight
            
    
    def forward(self, x: torch.Tensor, x_target: torch.Tensor):
        '''
        Run a sum of multiplications of all the losses
        '''
        
        loss_dict_values = {'total_loss': 0}
        for loss_type in self.cfg:
            loss_dict_values[loss_type] = self.loss_dict[loss_type](x, x_target)
            loss_dict_values['total_loss'] += loss_dict_values[loss_type] * self.loss_weight_dict[loss_type]
            
        return loss_dict_values
            
        