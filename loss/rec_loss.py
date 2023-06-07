from typing import Dict

import torch
import torch.nn.functional as F

from .base import LossBase, BASE_LOSS_TYPES


class RecLoss(LossBase):
    def __init__(self, loss_cfg: Dict):
        super().__init__(loss_cfg)
        self.use_tanh = False
        self.phase_parameter = 1

        if "use_tanh" in loss_cfg:
            if loss_cfg:
                self.use_tanh = True

        if "phase_parameter" in loss_cfg:
            self.phase_parameter = loss_cfg["phase_parameter"]

        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg["base_loss_type"]]

    def forward(self, x, x_target):
        pred_slice = x["output"]
        target_slice = x_target["music_slice"]

        if self.use_tanh:
            return self._phased_loss(
                F.tanh(pred_slice),
                F.tanh(target_slice),
                phase_parameter=self.phase_parameter,
            )
        else:
            return self._phased_loss(
                pred_slice, target_slice, phase_parameter=self.phase_parameter
            )

    def _phased_loss(
        self, x: torch.Tensor, x_target: torch.Tensor, phase_parameter: int = 10
    ):
        loss_vector = torch.zeros(phase_parameter * 2).to(x.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = self.base_loss_type(x, x_target)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
            else:
                loss_vector[idx * 2] = self.base_loss_type(
                    x[:, :, idx:], x_target[:, :, :-idx]
                )
                loss_vector[idx * 2 + 1] = self.base_loss_type(
                    x[:, :, :-idx], x_target[:, :, idx:]
                )
        return loss_vector.min()


class NoisePredLoss(LossBase):
    """
    Basic loss for reconstructing noise, used in diffusion
    """

    def __init__(self, loss_cfg: Dict):
        super().__init__(loss_cfg)
        self.base_loss_type = BASE_LOSS_TYPES[loss_cfg["base_loss_type"]]

    def forward(self, x, x_target):
        noise = x_target["noise"]
        noise_pred = x["noise_pred"]

        return self.base_loss_type(noise, noise_pred)
