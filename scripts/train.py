import argparse

from models.level_1_vqvae import Lvl1VQVariationalAutoEncoder
from utils.other import load_cfg_dict, initialize_trainer

def train_lvl_1_encoder(args):
    
    # Load model
    config_path = args.config if args.config is not None else 'config/lvl1_config.yaml'
    cfg = load_cfg_dict(config_path)
    model = Lvl1VQVariationalAutoEncoder(**cfg)
    if args.resume is not None:
        model = model.load_from_checkpoint(args.resume, **cfg)
        
    # Initialize trainer
    trainer = initialize_trainer(cfg, num_devices=args.num_devices)
    
    # Start training
    trainer.fit(model)
    


def main(args):
    
    choice = args.algorithm
    
    if choice == 'lvl1vqvae':
        train_lvl_1_encoder(args)
        
    elif choice == 'lvl2vqvae':
        raise NotImplementedError
    
    elif choice == 'lvl3vqvae':
        raise NotImplementedError
    
    else:
        raise ValueError(f'The algorithm type {choice} does not exist')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str, choices=['lvl1vqvae', 'lvl2vqvae', 'lvl3vqvae'],
                        help='The type of algorithm to train')
    parser.add_argument('-d', '--num_devices', type=int, default=1,
                        help='Number of GPU devices, 0 will use CPU.')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='The config file path. If none, the script will run the default config file according to the algorithm.')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='Model resume path, possible that the weights do not fit the model.')
    args = parser.parse_args()
    
    main(args)
    
    