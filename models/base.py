import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

import loaders


DATASETS = {'music_slice_dataset': loaders.MP3SliceDataset,
            'denoising_dataset': loaders.DenoiseDataset,
            'lvl2_dataset': loaders.Lvl2InputDataset,
            'lvl3_dataset': loaders.Lvl3InputDataset}


class BaseNetwork(pl.LightningModule):
    
    def __init__(self,
                 learning_rate: float,
                 weight_decay: float,
                 batch_size: int,
                 epochs: int,
                 dataset_name: str,
                 dataset_path: str,
                 scheduler_type: str,
                 eval_split_factor: float,
                 **kwargs) -> pl.LightningModule:
        
        super().__init__()
        
        # Initialize variables
        self.cfg = kwargs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.scheduler_name = scheduler_type
        
        # Datasets
        assert dataset_name in DATASETS, f'Dataset {dataset_name} is not in the datasets options.'
        assert 0 <= eval_split_factor <= 1, f'The split factor must be between 0 and 1, current value is {eval_split_factor}'
        self.dataset = None
        self.eval_split_factor = eval_split_factor
        self.dataset_name = dataset_name
        
        # Optimizers
        assert scheduler_type in ['none', 'one_cycle_lr', 'reduce_on_platou'] # TODO fix typo, program the schedulers in
        self.scheduler_type = scheduler_type
        
        
    def train_dataloader(self):
        self._set_dataset()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        self._set_dataset()
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def _set_dataset(self):
        
        if self.dataset is None:
            self.dataset = DATASETS[self.dataset_name](**self.cfg)
            train_dataset_length = int(len(self.dataset) * (1 - self.eval_split_factor))
            self.train_dataset, self.eval_dataset = random_split(self.dataset, 
                                                                (train_dataset_length, 
                                                                len(self.dataset) - train_dataset_length))
            
            
    def configure_optimizers(self):
        self._set_dataset()
        if len(self.dataset) % self.batch_size == 0:
            total_steps = len(self.dataset) // self.batch_size
        else:
            total_steps = len(self.dataset) // self.batch_size + 1
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'one_cycle_lr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, 
                                                            epochs=self.epochs,
                                                            steps_per_epoch=total_steps)
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1e6)
            
        elif self.scheduler_type == 'none':
            return [optimizer]
            
        scheduler_settings= {'scheduler': scheduler,
                             'interval': 'step',
                             'monitor': 'Training reconstruction loss',
                             'frequency': 1} # Set the scheduler to step every batch
        
        return [optimizer], [scheduler_settings]
    
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError('A forward method should be implemented')