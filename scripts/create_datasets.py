import argparse

import yaml

from models.multi_level_vqvae import MultiLvlVQVariationalAutoEncoder
import loaders
from loaders import MusicDataModule
from utils.other import load_cfg_dict


def main(args):
    device = args.device
    cfgs_vqvae = {
        1: "config/lvl1_config.yaml",
        2: "config/lvl2_config.yaml",
        3: "config/lvl3_config.yaml",
        4: "config/lvl4_config.yaml",
    }
    cfgs_gen = {
        2: "config/diff_lvl1_config.yaml",
        3: "config/diff_lvl2_config.yaml",
        4: "config/diff_lvl3_config.yaml",
        5: "config/diff_lvl4_config.yaml",
    }
    datasets = {
        1: loaders.MP3SliceDataset,
        2: loaders.Lvl2InputDataset,
        3: loaders.Lvl3InputDataset,
        4: loaders.Lvl4InputDataset,
    }
    vqvae_weights = {
        1: "model_weights/lvl1_vqvae.ckpt",
        2: "model_weights/lvl2_vqvae.ckpt",
        3: "model_weights/lvl3_vqvae.ckpt",
        4: "model_weights/lvl4_vqvae.ckpt",
    }

    # Generative flag
    init_idx = 2 if args.generative else 1
    cfgs = cfgs_gen if args.generative else cfgs_vqvae

    for idx in range(init_idx, 6):
        # Load the older model if necessary
        try:
            if idx > 1:
                cfg_prev = load_cfg_dict(cfgs_vqvae[idx - 1])
                prev_dataset = datasets[idx - 1](**cfg_prev, device=device)
                prev_vqvae = (
                    MultiLvlVQVariationalAutoEncoder.load_from_checkpoint(
                        vqvae_weights[idx - 1], **cfg_prev, strict=False
                    )
                    .to(device)
                    .eval()
                )
            else:
                cfg_prev = None
                prev_dataset = None
                prev_vqvae = None

        except ValueError:
            print(f"Could not load previous dataset and previous vqvae level {idx - 1}")
            cfg_prev = None
            prev_dataset = None
            prev_vqvae = None

        cfg_loaded = load_cfg_dict(cfgs[idx])
        del cfg_loaded["batch_size"]
        data_module = MusicDataModule(
            batch_size=1,
            latent_level=idx,
            previous_dataset=prev_dataset,
            previous_vqvae=prev_vqvae,
            **cfg_loaded,
        )
        data_module.setup("fit")

        # Print the sample sizes and the name of the track.
        loader = data_module.train_dataloader()
        for batch in loader:
            print(f"File name is {batch['track name']}")
            print(f"The size of the slice is {batch['music slice'].shape}")
            break

        del data_module, prev_dataset, prev_vqvae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Device for the vqvae."
    )
    parser.add_argument(
        "--generative",
        action="store_true",
        help="Flag for generating the generative datasets which have smaller sequence lengths.",
    )
    args = parser.parse_args()

    main(args)
