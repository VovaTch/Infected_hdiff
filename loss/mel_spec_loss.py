from typing import Dict

import torch
from torchaudio.transforms import MelSpectrogram

from .base import LossBase, BASE_LOSS_TYPES


class MelSpecLoss(LossBase):
    '''
    This loss is a reconstruction loss of a mel spectrogram, convert the inputs into a spectrogram and compute reconstruction loss
    '''
    
    def __init__(self, loss_cfg: Dict):
        super().__init__(loss_cfg)
        
        assert loss_cfg['base_loss_type'] in BASE_LOSS_TYPES, f'The base loss type'\
            ' must be one of: {[loss_type for loss_type in BASE_LOSS_TYPES]}'
        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg['base_loss_type']]
        self.mel_spec = MelSpectrogram(**loss_cfg['melspec_params'])
        
        self.lin_start = loss_cfg['lin_start']
        self.lin_end = loss_cfg['lin_end']
        
        
    def forward(self, x, x_target):
        
        pred_slice = x['output']
        target_slice = x_target['music_slice']
        
        self.mel_spec = self.mel_spec.to(pred_slice.device)
        
        return self.base_loss_type(self._mel_spec_and_process(pred_slice), self._mel_spec_and_process(target_slice))
        
        
    def _mel_spec_and_process(self, x: torch.Tensor):
        """
        To prepare the mel spectrogram loss, everything needs to be prepared.

        Args:
            x (torch.Tensor): Input, will be flattened
        """
        lin_vector = torch.linspace(self.lin_start, self.lin_end, self.loss_cfg['melspec_params']['n_mels'])
        eye_mat = torch.diag(lin_vector).to(x.device)
        mel_out = self.mel_spec(x.flatten(start_dim=0, end_dim=1))
        mel_out = torch.tanh(eye_mat @ mel_out)
        return mel_out