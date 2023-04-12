import argparse
import sys

from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from models.unet_denoiser import WaveUNet_Denoiser
from models.diffusion_vit import DiffusionViT
from utils.other import load_cfg_dict, initialize_trainer
from loss import TotalLoss
from loaders import MusicDataModule


# Check if we run in colab
IN_COLAB = 'google.colab' in sys.modules


def train_encoder(args, level: int=1):
    
    if IN_COLAB:
        print('Running on Google Colab.')
        
    # Load model with loss
    config_path_selection = {1: 'config/lvl1_config.yaml',
                             2: 'config/lvl2_config.yaml',
                             3: 'config/lvl3_config.yaml',
                             4: 'config/lvl4_config.yaml'}
    config_path = config_path_selection[level] if args.config is None else args.config
    cfg = load_cfg_dict(config_path)
    loss = TotalLoss(cfg['loss'])
    data_module = MusicDataModule(**cfg, latent_level=level)
    if args.resume is None:
        model = MultiLvlVQVariationalAutoEncoder(**cfg, loss_obj=loss)
    else:
        model = MultiLvlVQVariationalAutoEncoder.load_from_checkpoint(args.resume, **cfg, loss_obj=loss)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl{level}_vqvae.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
    

def train_denoiser(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
    
    # Load saved vqvae
    lvl1_config_path = 'config/lvl1_config.yaml'
    lvl1_cfg = load_cfg_dict(lvl1_config_path)
    lvl1_vqvae = MultiLvlVQVariationalAutoEncoder(**lvl1_cfg)
    lvl1_vqvae = lvl1_vqvae.load_from_checkpoint('model_weights/lvl1_vqvae.ckpt', **lvl1_cfg).requires_grad_(False)
    
    # Load model
    config_path = args.config if args.config is not None else 'config/denoiser_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = WaveUNet_Denoiser(**cfg, lvl1_vqvae=lvl1_vqvae)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, lvl1_vqvae=lvl1_vqvae, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/denoiser.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
        
        
def train_diff(args, level: int=0):
    
    if IN_COLAB:
        print('Running on Google Colab.')
        
    # Load model with loss
    config_path_selection = {0: 'config/denoiser_diff_config.yaml',
                             1: 'config/diff_lvl1_config.yaml',
                             2: 'config/diff_lvl2_config.yaml',
                             3: 'config/diff_lvl3_config.yaml',
                             4: 'config/diff_lvl4_config.yaml'}
    config_path = config_path_selection[level] if args.config is None else args.config
    cfg = load_cfg_dict(config_path)
    loss = TotalLoss(cfg['loss'])
    data_module = MusicDataModule(**cfg, latent_level=level + 1, dataset_cfg=cfg)
    if args.resume is None:
        model = DiffusionViT(**cfg, loss_obj=loss)
    else:
        model = DiffusionViT.load_from_checkpoint(args.resume, **cfg, loss_obj=loss)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
        
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/denoiser_diff.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
        

def main(args):
    
    choice = args.algorithm
    
    if choice == 'lvl1vqvae':
        train_encoder(args, level=1)
        
    elif choice == 'lvl2vqvae':
        train_encoder(args, level=2)
    
    elif choice == 'lvl3vqvae':
        train_encoder(args, level=3)
        
    elif choice == 'lvl4vqvae':
        train_encoder(args, level=4)
        
    elif choice == 'lvl1diff':
        train_diff(args, level=1)
        
    elif choice == 'lvl2diff':
        train_diff(args, level=2)
        
    elif choice == 'lvl3diff':
        train_diff(args, level=3)
        
    elif choice == 'lvl4diff':
        train_diff(args, level=4)
    
    elif choice == 'denoiser':
        train_denoiser(args)
        
    elif choice == 'denoiser_diff':
        train_diff(args, level=0)
    
    else:
        raise ValueError(f'The algorithm type {choice} does not exist')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, choices=['lvl1vqvae', 
                                                                'lvl2vqvae', 
                                                                'lvl3vqvae', 
                                                                'lvl4vqvae',
                                                                'lvl1diff',
                                                                'lvl2diff',
                                                                'lvl3diff',
                                                                'lvl4diff',
                                                                'denoiser', 
                                                                'denoiser_diff'],
                        help='The type of algorithm to train')
    parser.add_argument('-d', '--num_devices', type=int, default=1,
                        help='Number of GPU devices, 0 will use CPU.')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='The config file path. If none, the script will run the default config file according to the algorithm.')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Model resume path, possible that the weights do not fit the model.')
    args = parser.parse_args()
    
    main(args)
    
    