
import logging
import math
import os
import sys
from multiprocessing import Value  #a number shared btw num_workers
import torch
logger = logging.getLogger('ijepa')

def setup_logging():
    #prints messages with the time, e.g. "14:31:09 [1, 0/820] loss 0.45 ..."
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')


# weight initialization

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.   #cumulative distribution of N(0, 1), we can also use loops in
    # which we remove the number outside bounds, but that is uneffiecient

    with torch.no_grad():
        low, high = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()                  
        tensor.mul_(std * math.sqrt(2.))   
        tensor.add_(mean)                  
        tensor.clamp_(min=a, max=b)       
        return tensor


#masks

def apply_masks(x, masks):
    all_x = []
    for m in masks:
        # (B, K) -> (B, K, D), first the  feature is 1 then converted to dim size
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


#x is made of groups of B rows: [g0, g1]. repeats every group, like if we take repeat = 2, [g0, g0, g1, g1].
#with one context block (always the case here) repeat=1 and nothing changes.
def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    return torch.cat([torch.cat([x[i * B:(i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)], dim=0)


#the DataLoader calls this to turn a list of (image, label) into a batch. it also chooses the masks:
#the patches are numbered row by row: index = row * 14 + col for a 14 x 14 grid.
class MaskCollator:
    def __init__(
        self,
        input_size=224,
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),    #context block: 85 to 100% of the image (before the cut-outs)
        pred_mask_scale=(0.15, 0.2),   #each target block: 15-20% of the image
        aspect_ratio=(0.75, 1.5),      #target block height / width
        nenc=1,                        #number of context blocks
        npred=4,                       #number of target blocks
        min_keep=10,                   #a block must keep more than this many patches
        allow_overlap=False            
    ):
        self.height = self.width = input_size // patch_size   #14 for 224 / 16, 12 for 96 / 8
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        max_keep = int(self.height * self.width * (min_s + _rand * (max_s - min_s))) 
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        #h * w = max_keep and h / w = aspect_ratio
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        # shouldnt cross 14
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return h, w

    #acceptable_regions: grids with 0 where the block is not allowed (the target blocks).
    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        tries, timeout = 0, 20
        while True:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            if acceptable_regions is not None:
                #cut the targets out; after many failures, ignore the last `tries` targets
                for k in range(max(len(acceptable_regions) - tries, 0)):
                    mask *= acceptable_regions[k]
            mask = torch.nonzero(mask.flatten())   #indices of the kept patches, sorted
            if len(mask) > self.min_keep:
                break
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = 20
                logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top + h, left:left + w] = 0
        return mask.squeeze(), mask_complement

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)   #[images (B,3,H,W), labels (B)]
        g = torch.Generator()
        g.manual_seed(self.step())
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1., 1.))   #context is square
        #blockS positions are different for every image
        masks_pred, masks_enc = [], []
        min_keep_pred = min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            masks_enc.append(masks_e)

        #all masks in a batch must have the same length to be stacked into one tensor,
        #so every mask keeps only its first min_keep indices
        masks_pred = torch.utils.data.default_collate([[m[:min_keep_pred] for m in ms] for ms in masks_pred])
        masks_enc = torch.utils.data.default_collate([[m[:min_keep_enc] for m in ms] for ms in masks_enc])
        return collated_batch, masks_enc, masks_pred

# schedulers

def lr_at(step, t, ipe):
    warmup, total = int(t['warmup_epochs'] * ipe), int(t['epochs'] * ipe)
    if step < warmup:
        return t['start_lr'] + float(step) / float(max(1, warmup)) * (t['lr'] - t['start_lr'])
    progress = float(step - warmup) / float(max(1, total - warmup))
    return max(t['final_lr'], t['final_lr'] + (t['lr'] - t['final_lr']) * 0.5 * (1. + math.cos(math.pi * progress)))

def wd_at(step, t, ipe):
    wd0, wd1 = float(t['weight_decay']), float(t['final_weight_decay'])
    progress = step / int(t['epochs'] * ipe)
    wd = wd1 + (wd0 - wd1) * 0.5 * (1. + math.cos(math.pi * progress))
    return max(wd1, wd) if wd1 <= wd0 else min(wd1, wd)

def ema_at(i, t, ipe):
    m0, m1 = t['ema']
    return m0 + i * (m1 - m0) / (ipe * t['epochs'])

#ema update

@torch.no_grad()
def ema_update(encoder, target_encoder, m):
    for p, p_t in zip(encoder.parameters(), target_encoder.parameters()):
        p_t.data.mul_(m).add_((1. - m) * p.detach().data)

#checkpoints

#writes to a temporary file first and then renames it, so a crash while saving never leaves a half-written checkpoint
def save_checkpoint(path, **state):
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


#loads everything needed to continue training then returns the number of finished epochs
def load_checkpoint(
    path,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler
):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    predictor.load_state_dict(ckpt['predictor'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    optimizer.load_state_dict(ckpt['opt'])   #AdamW's running averages
    if scaler is not None and ckpt['scaler'] is not None:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt['epoch']
