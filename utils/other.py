import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from .ema import EMA

def load_cfg_dict(config_path):
    
    with open(config_path, 'r') as f:
        cfg_dict_raw = yaml.safe_load(f)
    cfg_dict = {**cfg_dict_raw['model'], **cfg_dict_raw['learning']}
    return cfg_dict


def initialize_trainer(cfg, num_devices: int=0) -> pl.Trainer:
    
    # Set device
    accelerator = 'cpu' if num_devices == 0 else 'gpu'
    num_devices = None if num_devices == 0 else num_devices
    
    # Configure trainer
    ema = EMA(cfg['beta_ema'])
    logger = TensorBoardLogger(name=cfg['name'], save_dir='saved/')
    model_checkpoint_callback = ModelCheckpoint(save_last=True, 
                                                save_weights_only=True, 
                                                save_top_k=1,
                                                monitor=cfg['monitored_loss'])
    model_summary = ModelSummary(max_depth=3)
    trainer = pl.Trainer(gradient_clip_val=cfg['gradient_clip'],
                         logger=logger,
                         callbacks=[ema, model_checkpoint_callback, model_summary],
                         devices=num_devices,
                         max_epochs=cfg['epochs'],
                         log_every_n_steps=1,
                         precision=16,
                         accelerator=accelerator)
    
    return trainer
