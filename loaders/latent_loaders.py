from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from .base_dataset import BaseLatentLoader


class Lvl2InputDataset(BaseLatentLoader):
    """
    This class takes the encoded level 1 vqvae latent vectors and collects them all in a .pt file, to be loaded for the level 2 training.
    The level 2 vqvae training will take 8 latents concatenated together as a sample.
    """
    
    def __init__(self,
                 collection_parameter: int=8,
                 prev_dataset=None,
                 prev_vqvae: 'MultiLvlVQVariationalAutoEncoder'=None,
                 slice_length: int=4096,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/001-datatensor.pt",
                 preload_metadata_file_path: str='data/music_samples/001-metadata.pkl',
                 **kwargs):
        
        super().__init__(collection_parameter, 
                         slice_length, 
                         preload_file_path, 
                         preload_metadata_file_path, 
                         device, prev_dataset, 
                         prev_vqvae, 
                         preload, 
                         **kwargs)
        
        
class Lvl3InputDataset(BaseLatentLoader):
    """
    This class takes the encoded level 2 vqvae latent vectors and collects them all in a .pt file, to be loaded for the level 2 training.
    The level 3 vqvae training will take 8 latents concatenated together as a sample.
    """
    
    def __init__(self,
                 collection_parameter: int=8,
                 prev_dataset: 'Lvl2InputDataset'=None,
                 prev_vqvae: 'MultiLvlVQVariationalAutoEncoder'=None,
                 slice_length: int=4096,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/002-datatensor.pt",
                 preload_metadata_file_path: str='data/music_samples/002-metadata.pkl',
                 **kwargs):
        
        super().__init__(collection_parameter, 
                         slice_length, 
                         preload_file_path, 
                         preload_metadata_file_path, 
                         device, prev_dataset, 
                         prev_vqvae, 
                         preload, 
                         **kwargs)
        
        
class Lvl4InputDataset(BaseLatentLoader):
    """
    This class takes the encoded level 3 vqvae latent vectors and collects them all in a .pt file, to be loaded for the level 3 training.
    The level 4 vqvae training will take 8 latents concatenated together as a sample.
    """
    
    def __init__(self,
                 collection_parameter: int=8,
                 prev_dataset: 'Lvl3InputDataset'=None,
                 prev_vqvae: 'MultiLvlVQVariationalAutoEncoder'=None,
                 slice_length: int=4096,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/003-datatensor.pt",
                 preload_metadata_file_path: str='data/music_samples/003-metadata.pkl',
                 **kwargs):
        
        super().__init__(collection_parameter, 
                         slice_length, 
                         preload_file_path, 
                         preload_metadata_file_path, 
                         device, prev_dataset, 
                         prev_vqvae, 
                         preload, 
                         **kwargs)
        
        
class Lvl5InputDataset(BaseLatentLoader):
    """
    This class takes the encoded level 4 vqvae latent vectors and collects them all in a .pt file, to be loaded for the highest level diffusion
    model to be trained.
    """
    
    def __init__(self,
                 collection_parameter: int=2,
                 prev_dataset: 'Lvl4InputDataset'=None,
                 prev_vqvae: 'MultiLvlVQVariationalAutoEncoder'=None,
                 slice_length: int=512,
                 device: str="cpu",
                 preload: bool=True,
                 preload_file_path: str="data/music_samples/004-datatensor.pt",
                 preload_metadata_file_path: str='data/music_samples/004-metadata.pkl',
                 **kwargs):
        
        super().__init__(collection_parameter, 
                         slice_length, 
                         preload_file_path, 
                         preload_metadata_file_path, 
                         device, prev_dataset, 
                         prev_vqvae, 
                         preload, 
                         **kwargs)