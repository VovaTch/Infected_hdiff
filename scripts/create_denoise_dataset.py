import argparse

from torch.utils.data import DataLoader

import loaders
from utils.other import load_cfg_dict


def main(args):
    device = args.device
    cfg_path = "config/denoiser_diff_config.yaml"
    cfg = load_cfg_dict(cfg_path)
    dataset = loaders.MP3SliceDataset(
        **cfg,
        device=device,
        preload_data_file_path="data/music_samples/000-datatensor_short.pt",
        preload_metadata_file_path="data/music_samples/000-metadata_short.pkl",
    )

    # Print the sample sizes and the name of the track.
    loader = DataLoader(dataset, batch_size=4)
    for batch in loader:
        print(f"File name is {batch['track name']}")
        print(f"The size of the slice is {batch['music slice'].shape}")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device for the vqvae."
    )
    args = parser.parse_args()

    main(args)
