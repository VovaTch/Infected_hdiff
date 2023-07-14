from typing import Dict

import torch

from .base import LossBase


class DecoderCrossEntropy(LossBase):
    def __init__(self, loss_cfg: Dict):
        """
        Standard cross-entropy-loss for training a decoder transformer for sequence generation.

        Args:
            loss_cfg (Dict): Loss configuration in the main configuration file
        """
        super().__init__(loss_cfg)
        self.cel_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self, x: Dict[str, torch.Tensor], x_target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward method for this configuration

        Args:
            x (Dict[str, torch.Tensor]): Dictionary expecting BS x V x cls in key "logits"
            x_target (Dict[str, torch.Tensor]): Dictionary expecting BS x V in key "latent indices"

        Returns:
            torch.Tensor: Cross entropy loss
        """
        logits = x["logits"][:-1]
        target_indices = x_target["latent indices"][1:]
        return self.cel_loss(logits, target_indices)
