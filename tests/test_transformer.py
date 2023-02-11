import torch
from torch.utils.data import DataLoader
import unittest

from models.diffusion_vit import DiffusionViT
from utils.other import load_cfg_dict
from loaders.music_loader import MP3SliceDataset

class TestDiT(unittest.TestCase):
    
    def setUp(self):
        
        # Load model
        cfg_path = 'config/denoiser_diff_config.yaml'
        self.cfg = load_cfg_dict(cfg_path)
        self.model = DiffusionViT(**self.cfg)
        self.cfg['batch_size'] = 1
        
        # Load dataset
        dataset = MP3SliceDataset()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Load batch
        for music_datapoint in self.dataloader:
            self.batch = music_datapoint['music slice']
            break
        
        super().setUp()
        
        
    def test_output(self):
        output = self.model(self.batch)
        self.assertEqual(output.shape, self.batch.shape)
        
        
    def test_conditional_output(self):
        output = self.model(self.batch, [self.batch, self.batch])
        self.assertEqual(output.shape, self.batch.shape)
        
        
if __name__ == "__main__":
    unittest.main()