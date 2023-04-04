import argparse

from torch.utils.data import DataLoader

from loaders.lvl3_loader import Lvl3InputDataset
from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
from utils.other import load_cfg_dict


def main(args):
    
    device = args.device
    
    # Load the config files
    cfg_2 = load_cfg_dict(args.config2)
    cfg_3 = load_cfg_dict(args.config3)
    
    # Load vqvae model
    vqvae = MultiLvlVQVariationalAutoEncoder(**cfg_2)
    vqvae = vqvae.load_from_checkpoint(args.resume, **cfg_2).to(device)
    vqvae._set_dataset()
    print(f'Loaded model from {args.resume}.')
    
    # Load the lvl2 dataset
    dataset = Lvl3InputDataset(**cfg_3, device=device, lvl2_dataset=vqvae.dataset, lvl2_vqvae=vqvae)
    loader = DataLoader(dataset, batch_size=4)
    
    # Print the sample sizes and the name of the track.
    for batch in loader:
        print(f"File name is {batch['track name']}")
        print(f"The size of the slice is {batch['music slice'].shape}")
        break
    
    print(f'Finished saving the dataset.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c2', '--config2', type=str, default='config/lvl2_config.yaml',
                        help='Config for the first level.')
    parser.add_argument('-c3', '--config3', type=str, default='config/lvl3_config.yaml',
                        help='Config for the second level.')
    parser.add_argument('-r', '--resume', type=str, default="model_weights/lvl2_vqvae.ckpt",
                        help='Path of the weights for the lvl2 vqvae model.')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device for the vqvae.')
    args = parser.parse_args()
    main(args)