import argparse

import torch

def train_lvl_1_encoder(args):
    
    config_path = args.config if args.config is not None else 'config/lvl1_config.yaml'


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
    
    