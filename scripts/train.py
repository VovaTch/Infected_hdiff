import argparse
import sys

# from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from models.multi_level_vqvae_new import MultiLvlVQVariationalAutoEncoder
from models.unet_denoiser import WaveUNet_Denoiser
from models.diffusion_vit import DiffusionViT
from utils.other import load_cfg_dict, initialize_trainer
from loss import TotalLoss


# Check if we run in colab
IN_COLAB = 'google.colab' in sys.modules


def train_encoder(args, level: int=1):
    
    if IN_COLAB:
        print('Running on Google Colab.')
        
    # Load model with loss
    config_path_selection = {1: 'config/lvl1_config_new.yaml',
                             2: '',
                             3: '',
                             4: ''}
    config_path = config_path_selection[level] if args.config is None else args.config
    cfg = load_cfg_dict(config_path)
    loss = TotalLoss(cfg['loss'])
    if args.resume is None:
        model = MultiLvlVQVariationalAutoEncoder(**cfg, loss_obj=loss)
    else:
        model = MultiLvlVQVariationalAutoEncoder.load_from_checkpoint(args.resume, **cfg, loss_obj=loss)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl{level}_vqvae.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
    


def train_lvl_1_encoder(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
    
    # Load model
    config_path = args.config if args.config is not None else 'config/lvl1_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = MultiLvlVQVariationalAutoEncoder(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl1_vqvae.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
        
        
def train_lvl_2_encoder(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
    
    # Load model
    config_path = args.config if args.config is not None else 'config/lvl2_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = MultiLvlVQVariationalAutoEncoder(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl2_vqvae.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
    
    
def train_lvl_3_encoder(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
    
    # Load model
    config_path = args.config if args.config is not None else 'config/lvl3_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = MultiLvlVQVariationalAutoEncoder(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl3_vqvae.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
        
        
def train_lvl_4_encoder(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
    
    # Load model
    config_path = args.config if args.config is not None else 'config/lvl4_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = MultiLvlVQVariationalAutoEncoder(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/lvl4_vqvae.ckpt'
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
        
        
def train_denoiser_diff(args):
    
    if IN_COLAB:
        print('Running on Google Colab.')
        
    # Load model
    config_path = args.config if args.config is not None else 'config/denoiser_diff_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = DiffusionViT(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg, strict=False)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
        
    # Start training
    trainer.fit(model)
    
    # If running on Colab
    if IN_COLAB:
        print('Saving checkpoint in Google Drive:')
        save_path = f'/content/drive/MyDrive/net_weights/IHDF/denoiser_diff.ckpt'
        trainer.save_checkpoint(save_path, weights_only=True)
        print(f'Saved network weights in {save_path}.')
        

def main(args):
    
    choice = args.algorithm
    
    # if choice == 'lvl1vqvae':
    #     train_lvl_1_encoder(args)
    
    if choice == 'lvl1vqvae':
        train_encoder(args, level=1)
        
    elif choice == 'lvl2vqvae':
        train_lvl_2_encoder(args)
    
    elif choice == 'lvl3vqvae':
        train_lvl_3_encoder(args)
        
    elif choice == 'lvl4vqvae':
        train_lvl_4_encoder(args)
    
    elif choice == 'denoiser':
        train_denoiser(args)
        
    elif choice == 'denoiser_diff':
        train_denoiser_diff(args)
    
    else:
        raise ValueError(f'The algorithm type {choice} does not exist')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, choices=['lvl1vqvae', 
                                                                'lvl2vqvae', 
                                                                'lvl3vqvae', 
                                                                'lvl4vqvae',
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
    
    