import torch
from torch.utils.data import DataLoader
import unittest

from models.diffusion_vit import DiffusionViT
from utils.other import load_cfg_dict
from loaders.lvl3_loader import Lvl3InputDataset

class TestDiT(unittest.TestCase):
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        
        # Load model
        cfg_path = 'config/denoiser_diff_config.yaml'
        self.cfg = load_cfg_dict(cfg_path)
        self.model = DiffusionViT(**self.cfg)
        self.cfg['batch_size'] = 1
        
        # Load dataset
        dataset = Lvl3InputDataset()
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    def setUp(self):
        
        # Load batch
        for music_datapoint in self.dataloader:
            self.batch = music_datapoint['music slice']
            break
        
        super().setUp()
        
        
    def test_output(self):
        output = self.model(self.batch, torch.tensor((3,)))
        self.assertEqual(output.shape, self.batch.shape)
        
        
    def test_conditional_output(self):
        output = self.model(self.batch, torch.tensor((5,)), [self.batch, self.batch])
        self.assertEqual(output.shape, self.batch.shape)
        
        
    def test_denoising(self):
        output = self.model.denoise(torch.randn_like(self.batch))
        self.assertEqual(output.shape, self.batch.shape)
        
        
    def test_denoising_conditional(self):
        output = self.model.denoise(torch.randn_like(self.batch), [self.batch, self.batch])
        self.assertEqual(output.shape, self.batch.shape)
        
    
    def test_patch_reconstruction(self):
        patched_batch = self.model._patchify(self.batch)
        reconstructed_batch = self.model._depatchify(patched_batch)
        self.assertEqual(torch.abs(self.batch - reconstructed_batch).sum().item(), 0)
        
        
if __name__ == "__main__":
    unittest.main()