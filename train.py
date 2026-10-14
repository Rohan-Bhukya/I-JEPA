import copy
import logging
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dataset import build_dataset, make_pretrain_loader, pretrain_transform
from utils import (MaskCollator, apply_masks, ema_at, ema_update, load_checkpoint, lr_at, repeat_interleave_batch, save_checkpoint, wd_at)
from vit import build_models

logger = logging.getLogger('ijepa')

def make_optimizer(encoder, predictor):
    """AdamW; weight decay only on weight matrices (not on biases / LayerNorm parameters).
    lr and weight decay are set every step from the schedules in utils.py."""
    def params(model, decay):
        return [p for n, p in model.named_parameters() if (('bias' not in n) and (len(p.shape) != 1)) == decay]
    return torch.optim.AdamW([{'params': params(encoder, True)}, {'params': params(predictor, True)},
                              {'params': params(encoder, False), 'WD_exclude': True, 'weight_decay': 0},
                              {'params': params(predictor, False), 'WD_exclude': True, 'weight_decay': 0}])


def set_lr_wd(optimizer, lr, wd):
    for g in optimizer.param_groups:
        g['lr'] = lr
        if not g.get('WD_exclude'):
            g['weight_decay'] = wd


#one training step

def train_step(encoder, predictor, target_encoder, optimizer, scaler, imgs, masks_enc, masks_pred, bf16):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        with torch.no_grad():
            h = target_encoder(imgs)                         # [B, N, D], the whole image
            h = F.layer_norm(h, (h.size(-1),))               # normalise each patch feature
            h = apply_masks(h, masks_pred)                   # [4B, K_pred, D] target blocks
            h = repeat_interleave_batch(h, len(imgs), repeat=len(masks_enc))
        z = encoder(imgs, masks_enc)                         # [B, K_enc, D] context only
        z = predictor(z, masks_enc, masks_pred)              # [4B, K_pred, D] predictions
        loss = F.smooth_l1_loss(z, h)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return float(loss.detach())

def train(cfg, device, resume=False, on_epoch_end=None):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    t, mk = cfg['train'], cfg['mask']
    bf16 = t['bf16'] and device.type == 'cuda'
    out = cfg['output']
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"{cfg['dataset']} | {cfg['model']}/{cfg['patch_size']} at {cfg['crop_size']}px | "
                f"predictor depth {cfg['predictor']['depth']} width {cfg['predictor']['width']} | {out}")

    encoder, predictor = build_models(cfg['model'], cfg['crop_size'], cfg['patch_size'],
                                      cfg['predictor']['depth'], cfg['predictor']['width'], device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    collator = MaskCollator(cfg['crop_size'], cfg['patch_size'], tuple(mk['context_scale']),
                            tuple(mk['target_scale']), tuple(mk['target_aspect_ratio']), nenc=1,
                            npred=mk['num_targets'], min_keep=mk['min_keep'], allow_overlap=mk['allow_overlap'])
    dataset = build_dataset(cfg['dataset'], cfg['data_root'], 'pretrain',
                            pretrain_transform(cfg['crop_size'], tuple(t['crop_scale'])))
    loader, sampler = make_pretrain_loader(dataset, collator, t['batch_size'], t['num_workers'],
                                           t['persistent_workers'])
    ipe = len(loader)
    logger.info(f'{len(dataset)} images, {ipe} iterations per epoch')

    optimizer = make_optimizer(encoder, predictor)
    scaler = torch.amp.GradScaler('cuda') if bf16 else None   # not needed for bf16, kept as in the original

    start_epoch = 0
    latest = os.path.join(out, 'jepa-latest.pth.tar')
    resume_path = resume if isinstance(resume, str) else (latest if resume else None)
    if resume_path and not os.path.exists(resume_path):
        if isinstance(resume, str):
            raise FileNotFoundError(f'--resume-from: {resume_path} does not exist')
        logger.warning(f'WARNING: --resume given but {latest} does not exist, starting from scratch. '
                       f'(The run folder name contains the number of epochs; to continue a run with a '
                       f'different epoch count use --resume-from <checkpoint>.)')
        resume_path = None
    if resume_path:
        start_epoch = load_checkpoint(resume_path, encoder, predictor, target_encoder, optimizer, scaler)
        collator._itr_counter.value = start_epoch * ipe - 1   # continue the mask seed sequence
        logger.info(f'resumed from {resume_path} after epoch {start_epoch}')

    csv_path = os.path.join(out, 'log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('epoch,itr,loss,context_patches,target_patches,ms\n')

    for epoch in range(start_epoch, t['epochs']):
        sampler.set_epoch(epoch)
        loss_sum, t0 = 0., time.time()
        with open(csv_path, 'a') as csv:
            for itr, ((imgs, _), masks_enc, masks_pred) in enumerate(loader):   # labels are never used
                step = epoch * ipe + itr
                lr, wd = lr_at(step + 1, t, ipe), wd_at(step + 1, t, ipe)
                set_lr_wd(optimizer, lr, wd)
                imgs = imgs.to(device, non_blocking=True)
                masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
                masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

                ts = time.time()
                loss = train_step(encoder, predictor, target_encoder, optimizer, scaler,
                                  imgs, masks_enc, masks_pred, bf16)
                ema_update(encoder, target_encoder, ema_at(step, t, ipe))
                ms = (time.time() - ts) * 1000
                if not math.isfinite(loss):
                    raise RuntimeError(f'loss is {loss} at epoch {epoch + 1}, iteration {itr}')

                loss_sum += loss
                csv.write(f'{epoch + 1},{itr},{loss:.5f},{masks_enc[0].shape[1]},{masks_pred[0].shape[1]},{ms:.0f}\n')
                if itr % t['log_every'] == 0:
                    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else 0
                    logger.info(f'[{epoch + 1}, {itr:5d}/{ipe}] loss {loss_sum / (itr + 1):.4f} | masks '
                                f'{masks_enc[0].shape[1]} {masks_pred[0].shape[1]} | lr {lr:.2e} wd {wd:.3f} | '
                                f'{mem:.0f} MB | {ms:.0f} ms')

        done = epoch + 1
        state = dict(encoder=encoder.state_dict(), predictor=predictor.state_dict(),
                     target_encoder=target_encoder.state_dict(), opt=optimizer.state_dict(),
                     scaler=None if scaler is None else scaler.state_dict(), epoch=done,
                     loss=loss_sum / ipe, config=cfg)
        save_checkpoint(latest, **state)
        if done % t['checkpoint_every'] == 0 or done == t['epochs']:
            save_checkpoint(os.path.join(out, f'jepa-ep{done}.pth.tar'), **state)
        logger.info(f'epoch {done} done: avg loss {loss_sum / ipe:.4f}, {(time.time() - t0) / 60:.1f} min')
        if on_epoch_end is not None:
            on_epoch_end()
    return latest
