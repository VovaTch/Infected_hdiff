import torch
from torch.utils.data import DataLoader
import unittest

from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from utils.other import load_cfg_dict
from loaders.music_loader import MP3SliceDataset

class TestLvl1VQVAE_cpu(unittest.TestCase):
    
    def setUp(self):
        
        # Load model
        cfg_path = 'config/lvl1_config.yaml'
        self.cfg = load_cfg_dict(cfg_path)
        self.model = MultiLvlVQVariationalAutoEncoder(**self.cfg)
        self.cfg['batch_size'] = 1
        
        # Load dataset
        dataset = MP3SliceDataset()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Load batch
        for music_datapoint in self.dataloader:
            self.batch = music_datapoint['music slice']
            break
        
        super().setUp()
        
    def test_encoder(self):
        encoder_output = self.model.encoder(self.batch)
        self.assertEqual(encoder_output.shape[0], 1)
        self.assertEqual(encoder_output.shape[1], self.cfg['latent_depth'])
        
    def test_vq(self):
        encoder_output = self.model.encoder(self.batch)
        vq_output = self.model.vq_module(encoder_output)
        self.assertEqual(encoder_output.size(), vq_output['v_q'].size())
        
    def test_vqvae(self):
        total_output = self.model(self.batch)
        self.assertEqual(self.batch.size(), total_output['output'].size())
        
        
class TestLvl1VQVAE_gpu(unittest.TestCase):
    
    def setUp(self):
        
        # Load model
        cfg_path = 'config/lvl1_config.yaml'
        self.cfg = load_cfg_dict(cfg_path)
        self.model = MultiLvlVQVariationalAutoEncoder(**self.cfg).to(device='cuda')
        self.cfg['batch_size'] = 1
        
        # Load dataset
        dataset = MP3SliceDataset()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Load batch
        for music_datapoint in self.dataloader:
            self.batch = music_datapoint['music slice'].to(device='cuda')
            break
        
        super().setUp()
        
    def test_encoder(self):
        encoder_output = self.model.encoder(self.batch)
        self.assertEqual(encoder_output.shape[0], 1)
        self.assertEqual(encoder_output.shape[1], self.cfg['latent_depth'])
        
    def test_vq(self):
        encoder_output = self.model.encoder(self.batch)
        vq_output = self.model.vq_module(encoder_output)
        self.assertEqual(encoder_output.size(), vq_output['v_q'].size())
        
    def test_vqvae(self):
        total_output = self.model(self.batch)
        self.assertEqual(self.batch.size(), total_output['output'].size())
        
        
if __name__ == "__main__":
    unittest.main()