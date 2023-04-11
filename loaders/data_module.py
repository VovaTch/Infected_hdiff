import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .music_loader import MP3SliceDataset
from .lvl2_loader import Lvl2InputDataset
from .lvl3_loader import Lvl3InputDataset
from .lvl4_loader import Lvl4InputDataset


DATASETS = {1: MP3SliceDataset,
            2: Lvl2InputDataset,
            3: Lvl3InputDataset,
            4: Lvl4InputDataset}

class MusicDataModule(pl.LightningDataModule):
    """
    This data module decouples the model with the datasets. 
    """
    
    def __init__(self, 
                 batch_size: int, 
                 latent_level: int=1, 
                 eval_split_factor: float=0.01,
                 previous_dataset=None, 
                 previous_vqvae=None,
                 data_path: str='data/music_samples/',
                 **kwargs):
        
        super().__init__()
        assert latent_level in [1, 2, 3, 4], 'The latent level must be from 1 to 4.'
        self.latent_level = latent_level
        self.previous_dataset = previous_dataset # Previous dataset in case the data needs to be created
        self.previous_vqvae = previous_vqvae # Previous vqvae in case the data needs to be created
        self.batch_size = batch_size
        self.eval_split_factor = eval_split_factor
        self.data_path = data_path
        
        
    def setup(self, stage: str):
        
        if stage == 'fit':
            
            dataset = DATASETS[self.latent_level](prev_dataset=self.previous_dataset, 
                                                  prev_vqvae=self.previous_vqvae,
                                                  audio_dir=self.data_path)
            train_dataset_length = int(len(dataset) * (1 - self.eval_split_factor))
            self.train_dataset, self.eval_dataset = random_split(dataset, 
                                                                 (train_dataset_length, 
                                                                 len(dataset) - train_dataset_length))
            
            self.total_dataset_len = len(dataset)
            
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    
    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def get_train_dataset_length(self):
        return len(self.total_dataset_len) // self.batch_size if\
            len(self.total_dataset_len) % self.batch_size == 0 else\
            len(self.total_dataset_len) // self.batch_size + 1
            