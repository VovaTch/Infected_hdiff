import argparse
from typing import OrderedDict
import random

import yaml

from integrator import InfectedHDiffIntegrator


def main(args):
    
    # Load the config
    device = args.device
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    module_cfgs = OrderedDict(cfg['models'])
    
    # Load and run the integrator
    integrator = InfectedHDiffIntegrator(modules=module_cfgs, device=device, **cfg)
    track_length = args.min_time + random.random() * (args.max_time - args.min_time)
    integrator(track_length=track_length)
    
    # Save the track
    integrator.save_track(path=args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Type of device to run inference on.")
    parser.add_argument("-c", "--config", type=str, default="config/integrator_config.yaml",
                        help='Config path of the integrator.')
    parser.add_argument("-s", "--save_path", type=str, default="src/output/sound/sample.mp3",
                        help="Save path of the resulting MP3 file.")
    parser.add_argument("--min_time", type=float, default=270,
                        help="Minimum track time in seconds.")
    parser.add_argument("--max_time", type=float, default=600,
                        help="Maximum track time in seconds.")
    args = parser.parse_args()
    
    main(args)