import argparse

import torch

from models.level_1_vqvae import Lvl1VQVariationalAutoEncoder
from utils.other import load_cfg_dict
import loaders


def main(args):
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = args.config if args.config is not None else 'config/lvl1_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = Lvl1VQVariationalAutoEncoder(**cfg).to(device=device)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False).to(device=device)
        
    lvl1_dataset = loaders.MP3SliceDataset()
    _ = loaders.DenoiseDataset(model, lvl1_dataset)
    print('Data .pkl file was created.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='data/',
                        help='Path for the pickle file to be created.')
    parser.add_argument('-r', '--resume', type=str, default='model_weights/lvl1_vqvae.ckpt',
                        help='Weights path for the lvl1 vqvae')
    parser.add_argument('-c', '--config', type=str, default='config/lvl1_config.yaml',
                        help='Path to the config file of the lvl1 vqvae')
    args = parser.parse_args()
    main(args)