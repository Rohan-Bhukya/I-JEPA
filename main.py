import argparse
import pprint
import torch
import yaml
from train import train
from utils import logger, setup_logging

def apply_overrides(config, overrides):
    for item in overrides:
        dotted_key, value = item.split('=', 1)# lets say data.branch_size=128, data.brach_size becomes key
        node = config
        *parents, last = dotted_key.split('.')#data is parent
        for k in parents:
            node = node[k]
        node[last] = yaml.safe_load(value)# we will change it with 128
    return config

def main():
    parser = argparse.ArgumentParser(description='I-JEPA pre-training')
    parser.add_argument('--config', default='config.yaml', help='YAML config file')
    parser.add_argument('--device', default=None,help='torch device, e.g. cuda:0 or cpu (default: cuda:0 if available)')
    parser.add_argument('--resume', action='store_true',help='resume from the latest checkpoint in logging.folder (if it exists)')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',help='override config entries, e.g. --set data.batch_size=64')
    args = parser.parse_args()
    setup_logging()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_overrides(config, args.set)
    if args.resume:
        config['meta']['load_checkpoint'] = True
    logger.info('loaded params...\n' + pprint.pformat(config, indent=4))
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    logger.info(f'device: {device}')
    train(config, device)

if __name__ == '__main__':
    main()
