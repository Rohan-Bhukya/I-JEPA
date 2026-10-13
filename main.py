import argparse
import pprint
import torch
import yaml
from dataset import DATASETS
from train import train
from utils import logger, setup_logging
from vit import PREDICTOR, VIT_SIZES

def load_config(
    path,
    dataset=None,
    model=None,
    overrides=()
):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if dataset:
        cfg['dataset'] = dataset
    if model:
        cfg['model'] = model
    for item in overrides:
        key, value = item.split('=', 1)#lets say train.epochs=300, train.epochs becomes key
        *parents, last = key.split('.')#train is parent
        node = cfg
        for k in parents:
            node = node[k]
        node[last] = yaml.safe_load(value)#we will change it with 300

    if cfg['dataset'] not in DATASETS:
        raise ValueError(f"unknown dataset {cfg['dataset']!r}, choose from {list(DATASETS)}")
    if cfg['model'] not in VIT_SIZES:
        raise ValueError(f"unknown model {cfg['model']!r}, choose from {list(VIT_SIZES)}")
    #if not given in config, take crop and patch size from the dataset
    cfg['crop_size'] = cfg['crop_size'] or DATASETS[cfg['dataset']]['crop_size']
    cfg['patch_size'] = cfg['patch_size'] or DATASETS[cfg['dataset']]['patch_size']
    if cfg['crop_size'] % cfg['patch_size']:
        raise ValueError(f"crop_size {cfg['crop_size']} is not a multiple of patch_size {cfg['patch_size']}")
    depth, width = PREDICTOR[cfg['model']]
    cfg['predictor']['depth'] = cfg['predictor']['depth'] or depth
    cfg['predictor']['width'] = cfg['predictor']['width'] or width
    #output folder name, like runs/imagenet100_vits16_ep100
    arch = f"vit{cfg['model'].split('_')[1][0]}{cfg['patch_size']}"
    cfg['output'] = cfg['output'].format(dataset=cfg['dataset'], arch=arch, epochs=cfg['train']['epochs'])
    return cfg


def main(argv=None, on_epoch_end=None):
    parser = argparse.ArgumentParser(description='I-JEPA pre-training')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--dataset', choices=list(DATASETS))
    parser.add_argument('--model', choices=list(VIT_SIZES))
    parser.add_argument('--resume', action='store_true', help='continue from <output>/jepa-latest.pth.tar')
    parser.add_argument('--resume-from', metavar='CHECKPOINT',
                        help='continue from this checkpoint, e.g. to extend a 100-epoch run to 300 epochs '
                             '(the lr / wd / EMA schedules then follow the new number of epochs)')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='e.g. train.epochs=300')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config, args.dataset, args.model, args.set)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if os.name == 'nt':#on windows, limit how much gpu memory we use
            torch.cuda.set_per_process_memory_fraction(cfg['train']['gpu_memory_fraction'], device)
    return train(cfg, device, resume=args.resume_from or args.resume, on_epoch_end=on_epoch_end)


if __name__ == '__main__':
    main()
