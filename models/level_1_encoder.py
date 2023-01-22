from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

class Lvl1Encoder(pl.LightningModule):
    
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        pass
    
    