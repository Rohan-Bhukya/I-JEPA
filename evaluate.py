#Evaluate.py

import argparse
import itertools
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataset import DATASETS, build_dataset, make_eval_loader, probe_train_transform, test_transform
from utils import setup_logging, trunc_normal_
from vit import make_encoder

logger = logging.getLogger('ijepa')


def load_target_encoder(path, device):
## make_encoder correct architecture --> load the saved weights ---> load appropriate parameter names --> freeze parameters --> return target encode in eval mode
    ckpt = torch.load(path, map_location='cpu', weights_only=False) # path runs/jepa-latest.pth.tar
    cfg = ckpt['config']
    encoder = make_encoder(cfg['model'], cfg['crop_size'], cfg['patch_size']) # building a neural network structure  and load the learned weights into that structure
    # for k,v in ckpt['target_encoder'].items():
    #    print(k)      k.removeprefix('module.') removes prefixes such as in module.blocks.0.attn.qkv.weight because new model doesn't expect it
    # dict making with key and value v
    encoder.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt['target_encoder'].items()})
    for p in encoder.parameters():
        p.requires_grad = False # freeze  the encoder parameters can't be modified by gradient descent
    logger.info(f"loaded target encoder: {cfg['model']}/{cfg['patch_size']}, trained on {cfg['dataset']} "
                f"for {ckpt['epoch']} epochs ({path})")
    return encoder.to(device).eval(), cfg


@torch.no_grad() # no grad tracking inside the function

def features(encoder, imgs, bf16):

# imgs ---> patch_embedd + pos_embedd --> transformer block output ---> normalize the average of last 4 layers
# we use bfloat16 during computation to reduce GPU memory usage and potentially speed up processing then to float for precise calculations
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        x = encoder.patch_embed(imgs) + encoder.pos_embed
        outs = []
        for blk in encoder.blocks:
            x = blk(x)# output on 1 block is input of other
            outs.append(x) # storing every output because we need to extract representations from all transformer
        # average pooling over patch tokens
        pooled = [encoder.norm(o).float().mean(dim=1) for o in outs[-4:]]   # final LayerNorm on each
        # for last 4 outputs of tranformer block mean is taken along dim 1 (B, N, D)-->(B,D)
        # data type converted to float32 and outs are normalized earlier

    return pooled[-1], torch.cat(pooled, dim=1) # torch.cat (B,D)-->(B,4D)



@torch.no_grad()
def extract(encoder, dataset, batch_size, num_workers, device, bf16):
    # extract() repeatedly calls features() for every batch in the dataset, saves the outputs, and combines them into complete feature tensors.
    last, last4, labels = [], [], []
    for imgs, y in make_eval_loader(dataset, batch_size, False, num_workers):
        a, b = features(encoder, imgs.to(device, non_blocking=True), bf16)
        last.append(a)
        last4.append(b)
        labels.append(y.to(device))
    return torch.cat(last), torch.cat(last4), torch.cat(labels)


# -- kNN
#k-Nearest Neighbors (k-NN) checks whether images with similar learned embeddings belong to the same class.

@torch.no_grad()
def knn_accuracy(train_feats, train_labels, test_feats, test_labels, num_classes, k=20, temperature=0.07):
    
    # train_feats : embeddings of training images
    # k=20: use the 20 most similar training images to classify each test image.
    #temperature=0.07: controls how strongly similarity affects each neighbor's vote.
    
    train_feats, test_feats = F.normalize(train_feats, dim=1), #F.normalize(test_feats, dim=1) # Normalizes every feature vector to have unit length , so similarity comparisons focus on direction of embeddings 

    k = min(k, len(train_feats))   # k never exceeds the number of availabele training examples . if there are 10 neighbours instead of 20
    correct = 0
    for i in range(0, len(test_feats), 256): # processing test image in batches of 256
        
        sims, idx = (test_feats[i:i + 256] @ train_feats.T).topk(k, dim=1) # cosine similarity 
        votes = torch.zeros(len(sims), num_classes, device=sims.device) 
        votes.scatter_add_(1, train_labels[idx], (sims / temperature).exp())   # similarity-weighted vote
        correct += (votes.argmax(1) == test_labels[i:i + 256]).sum().item()
    return 100. * correct / len(test_feats)


# -- linear probe

## LARS is reused from MAE's linear -probing implementation for our I-JEPA evaluation . LARS is optimizer
class LARS(torch.optim.Optimizer):

    #One-dimensional parameters, usually biases, receive no weight decay.
    #One-dimensional parameters also receive no trust-ratio scaling.
    
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,  #lr = 0 no learning rate 
                                      trust_coefficient=trust_coefficient))

# gradients from backprop p.grad ---> Add weight decay ---> calculate trust ratio ---> Apply momentum --->Update classifier parameters

    @torch.no_grad()
    # optimizer changes parameters directly, so we disable gradient tracking while performing those updates
    def step(self):

        for g in self.param_groups:  
            for p in g['params']:
                dp = p.grad 
                if dp is None:
                    continue
                if p.ndim > 1:
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm, update_norm = torch.norm(p), torch.norm(dp)
                    #param_norm: magnitude of the parameter tensor.
                    # update_norm: magnitude of its adjusted gradient.
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0., torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)
                state = self.state[p]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(p)
                state['mu'].mul_(g['momentum']).add_(dp)
                p.add_(state['mu'], alpha=-g['lr'])

# make_head creates linear classifier that we train during linear probing while I-JEPA encoder frozen

def make_head(dim, num_classes, batchnorm):
    linear = nn.Linear(dim, num_classes)
    trunc_normal_(linear.weight, std=0.01)        # MAE linear-probe initialisation
    nn.init.zeros_(linear.bias)
    return nn.Sequential(nn.BatchNorm1d(dim, affine=False, eps=1e-6), linear) if batchnorm else linear


def linear_probe(encoder, e, train_set, test_set, num_classes, device, bf16, num_workers):

    # pre- extracted I-JEPA features --> Create multiple linear heads ---> Configure LARS and learning ratee milestones ---> train & evaluate heads 
    batch = e['probe_batch_size'] or min(16384, max(64, round(len(train_set) / 78))) # if probe_batch_size 
    batch = min(batch, len(train_set))           # at least one step per epoch
    epochs = e['probe_epochs']
    milestones = [round(epochs * f) for f in (0.3, 0.6, 0.9)]     # 15, 30, 45 for 50 epochs learning rate will reduce by factor of 10 at these milestones 
    dim = encoder.embed_dim
    # combes makes every possible combination of these choices . no of comb 2*2*2*2 =16

    combos = list(itertools.product(['last layer', 'last 4 layers'], [False, True],
                                    e['probe_lrs'], e['probe_weight_decays']))
    heads = nn.ModuleList([make_head(dim if rep == 'last layer' else 4 * dim, num_classes, bn)  # for lat layer dim else 4*dim  4 feature vectors concatenated
                           for rep, bn, _, _ in combos]).to(device) # rep - representation (last layer / last 4 layer) bn indicates batch norm enabled(True)
    
    base_lrs = [blr * 16384 / 256 for _, _, blr, _ in combos]    # the paper's absolute learning rates
    optimizer = LARS([{'params': h.parameters(), 'lr': lr, 'weight_decay': wd}
                      for h, lr, (_, _, _, wd) in zip(heads, base_lrs, combos)])
    
    logger.info(f'linear probe: {len(combos)} heads, batch {batch}, {epochs} epochs, '
                f'lr / 10 at epochs {milestones}, {len(train_set) // batch} steps per epoch')


# frozen I-JEPA encoder --> train multiple linear heads --> Evaluate on test features --> save the best head

    def step(last, last4, y):
        # heads are independent, so one backward of the summed losses trains each on its own loss combined loss helps for backward pass and calculate gradients for every classifer
        loss = sum(F.cross_entropy(h(last if rep == 'last layer' else last4), y) # each classifier updates it loss
                   for h, (rep, _, _, _) in zip(heads, combos))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for epoch in range(epochs):
        factor = 0.1 ** sum(epoch >= m for m in milestones)
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g['lr'] = lr * factor
        heads.train()
        t0, buf = time.time(), []
        for imgs, y in make_eval_loader(train_set, e['batch_size'], True, num_workers):
            a, b = features(encoder, imgs.to(device, non_blocking=True), bf16)
            buf.append((a, b, y.to(device)))
            if sum(len(x[2]) for x in buf) >= batch:      # collect frozen features up to one probe batch
                a, b, y = (torch.cat(t) for t in zip(*buf))
                step(a[:batch], b[:batch], y[:batch])
                buf = [(a[batch:], b[batch:], y[batch:])]
        # what is left in buf at the end of the epoch (< one batch) is dropped, like drop_last=True in MAE
        logger.info(f'probe epoch {epoch + 1}/{epochs}, lr x{factor:g} ({time.time() - t0:.0f} s)')

    test_last, test_last4, test_y = extract(encoder, test_set, e['batch_size'], num_workers, device, bf16)
    heads.eval()
    results = []
    with torch.no_grad():
        for i, (h, (rep, bn, blr, wd)) in enumerate(zip(heads, combos)):
            pred = h(test_last if rep == 'last layer' else test_last4).argmax(1)
            results.append((100. * (pred == test_y).float().mean().item(), rep, 'BN + linear' if bn else 'linear', blr, wd, i))
    results.sort(key=lambda r: r[0], reverse=True)
    best = results[0]
    # the best head, so inference.py can classify new images with it
    best_head = {'state_dict': heads[best[5]].state_dict(), 'representation': best[1], 'batchnorm': combos[best[5]][1],
                 'dim': dim if best[1] == 'last layer' else 4 * dim, 'top1': best[0]}
    return [r[:5] for r in results], best_head


def main(argv=None):
    # argparse lets you customize how evaluate.py runs through terminal arguments, instead of changing the source code every time.
    parser = argparse.ArgumentParser(description='kNN and linear-probe evaluation of an I-JEPA checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=list(DATASETS), help='default: the dataset the model was trained on')
    parser.add_argument('--knn', action='store_true', help='only kNN (default: kNN and linear probe)')
    parser.add_argument('--probe', action='store_true', help='only the linear probe')
    parser.add_argument('--probe-epochs', type=int, help='default: eval.probe_epochs (50)')
    parser.add_argument('--probe-out', help='where to save the best head (default: linear_probe_<dataset>.pth '
                                            'next to the checkpoint)')
    parser.add_argument('--config', default='config.yaml', help='eval settings are read from here if it exists')
    parser.add_argument('--data-root', help='default: data_root of the training config')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    setup_logging()
    if not (args.knn or args.probe):
        args.knn = args.probe = True

    device = torch.device(args.device)
    bf16 = device.type == 'cuda'
    encoder, cfg = load_target_encoder(args.checkpoint, device)
    e = dict(cfg['eval'])
    if os.path.exists(args.config):
        with open(args.config) as f:
            e.update(yaml.safe_load(f).get('eval') or {})
    if args.probe_epochs:
        e['probe_epochs'] = args.probe_epochs
    name = args.dataset or cfg['dataset']
    root = args.data_root or cfg['data_root']
    crop = cfg['crop_size']          # images are fed at the resolution the model was trained at
    test_set = build_dataset(name, root, 'eval_test', test_transform(crop))
    num_classes = len(test_set.classes)
    logger.info(f'evaluating on {name}: {num_classes} classes, {len(test_set)} test images, {crop}px')
    results = {}

    if args.knn:
        train_set = build_dataset(name, root, 'eval_train', test_transform(crop))
        tr, _, tr_y = extract(encoder, train_set, e['batch_size'], args.num_workers, device, bf16)
        te, _, te_y = extract(encoder, test_set, e['batch_size'], args.num_workers, device, bf16)
        results['knn'] = knn_accuracy(tr, tr_y, te, te_y, num_classes, e['knn_k'], e['knn_temperature'])
        logger.info(f"kNN (k={e['knn_k']}) top-1: {results['knn']:.2f}%")

    if args.probe:
        train_set = build_dataset(name, root, 'eval_train', probe_train_transform(crop))
        table, best_head = linear_probe(encoder, e, train_set, test_set, num_classes, device, bf16, args.num_workers)
        logger.info('linear probe, all combinations (top-1 %, representation, head, reference lr, weight decay):')
        for acc, rep, head, blr, wd in table:
            logger.info(f'  {acc:6.2f}  {rep:13s}  {head:11s}  lr {blr:<6}  wd {wd}')
        best = table[0]
        results['linear_probe'] = best[0]
        logger.info(f'linear probe top-1 (best of {len(table)} combinations, chosen on the test set as in the paper): {best[0]:.2f}% '
                    f'({best[1]}, {best[2]}, lr {best[3]}, wd {best[4]})')
        probe_path = args.probe_out or os.path.join(os.path.dirname(args.checkpoint), f'linear_probe_{name}.pth')
        torch.save({**best_head, 'classes': list(test_set.classes), 'dataset': name,
                    'checkpoint': os.path.abspath(args.checkpoint)}, probe_path)
        logger.info(f'saved the best head to {probe_path} (used by inference.py --probe)')
    print(results)
    return results


if __name__ == '__main__':
    main()

