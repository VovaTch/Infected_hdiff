from typing import Dict

import torch.nn.functional as F

from .base import LossBase, BASE_LOSS_TYPES


class RecLoss(LossBase):
    
    def __init__(self, loss_cfg: Dict):
        super().__init__(loss_cfg)
        self.use_tanh = False
        if 'use_tanh' in loss_cfg:
            if loss_cfg:
                self.use_tanh = True
                
        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg['base_loss_type']]
                
                
    def forward(self, x, x_target):
        
        pred_slice = x['output']
        target_slice = x_target['music_slice']
        
        if self.use_tanh:
            return self.base_loss_type(F.tanh(pred_slice), F.tanh(target_slice))
        else:
            return self.base_loss_type(pred_slice, target_slice)