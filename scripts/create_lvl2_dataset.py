import argparse

from torch.utils.data import DataLoader

from loaders.lvl2_loader import Lvl2InputDataset
from models.level_1_vqvae import Lvl1VQVariationalAutoEncoder
from utils.other import load_cfg_dict


def main(args):
    
    device = args.device
    
    # Load the config files
    cfg_1 = load_cfg_dict(args.config1)
    cfg_2 = load_cfg_dict(args.config2)
    
    # Load vqvae model
    vqvae = Lvl1VQVariationalAutoEncoder(**cfg_1)
    vqvae._set_dataset()
    vqvae = vqvae.load_from_checkpoint(args.resume, **cfg_1).to(device)
    print(f'Loaded model from {args.resume}.')
    
    # Load the lvl2 dataset
    dataset = Lvl2InputDataset(**cfg_2, device=device, lvl1_dataset=vqvae.dataset, lvl1_vqvae=vqvae)
    loader = DataLoader(dataset, batch_size=4)
    
    # Print the sample sizes and the name of the track.
    for batch in loader:
        print(f"File name is {batch['track name']}")
        print(f"The size of the slice is {batch['lvl2 slice'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config1', type=str, default='config/lvl1_config.yaml',
                        help='Config for the first level.')
    parser.add_argument('-c2', '--config2', type=str, default='config/lvl2_config.yaml',
                        help='Config for the second level.')
    parser.add_argument('-r', '--resume', type=str, default='model_weights/lvl1_vqvae.ckpt',
                        help='Path of the weights for the lvl1 vqvae model.')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device for the vqvae.')
    args = parser.parse_args()
    main(args)