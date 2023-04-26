import math

import torch
import torch.nn as nn
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from .ema import EMA

def load_cfg_dict(config_path):
    
    with open(config_path, 'r') as f:
        cfg_dict_raw = yaml.safe_load(f)
    cfg_dict = {**cfg_dict_raw['model'], **cfg_dict_raw['learning']}
    del cfg_dict_raw['model'], cfg_dict_raw['learning']
    cfg_dict.update(**cfg_dict_raw)
    return cfg_dict


def initialize_trainer(cfg, num_devices: int=0) -> pl.Trainer:
    
    # Set device
    accelerator = 'cpu' if num_devices == 0 else 'gpu'
    num_devices = None if num_devices == 0 else num_devices
    
    # Configure trainer
    ema = EMA(cfg['beta_ema'])
    learning_rate_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(name=cfg['name'], save_dir='saved/')
    model_checkpoint_callback = ModelCheckpoint(save_last=True, 
                                                save_weights_only=True, 
                                                save_top_k=1,
                                                monitor=cfg['monitored_loss'])
    
    # AMP 
    if 'amp' in cfg:
        if cfg['amp']:
            precision = 16
        else:
            precision = 32
    else:
        precision = 32
    
    model_summary = ModelSummary(max_depth=3)
    trainer = pl.Trainer(gradient_clip_val=cfg['gradient_clip'],
                         logger=logger,
                         callbacks=[model_checkpoint_callback, model_summary, learning_rate_monitor, ema],
                         devices=num_devices,
                         max_epochs=cfg['epochs'],
                         log_every_n_steps=1,
                         precision=precision,
                         accelerator=accelerator)
    
    return trainer


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.stack((embeddings.sin(), embeddings.cos()), dim=2).view(-1, self.dim)
        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings
    
    
def getPositionEncoding(seq_len: int, d: int, n: int=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2 * i] = torch.sin(k / denominator)
            P[k, 2 * i + 1] = torch.cos(k / denominator)
    return P
