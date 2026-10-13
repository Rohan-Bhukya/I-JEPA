#vit has 2 parts here
#1. the encoder (we use it twice, context encoder and target encoder)
#2. the predictor which predicts the target block representations from the context

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import trunc_normal_, apply_masks, repeat_interleave_batch


#positional encoding
#transformer has no idea where a patch is in the img, so we add sin/cos values based on position
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)#every position * every frequency
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    #for 2d we do 1d for rows and 1d for cols, each gets half of embed_dim
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

#transformer block

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5#1/sqrt(d), keeps the scores from getting too big
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)#q,k,v in one go
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        #(B,N,3C) -> (3,B,heads,N,C/heads) so we can split q,k,v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale#how much each patch looks at every other patch
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#merge heads back
        return self.proj(x)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, norm_layer):
        super().__init__()
        #keep this order, changing it changes the random init
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):#converts img into patch vectors

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)#224/16 = 14, so 196 patches
        #conv with kernel = stride = patch size is same as cutting into patches + linear layer
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #(B,D,14,14) -> (B,196,D)
        return self.proj(x).flatten(2).transpose(1, 2)

#weight init
def _init_weights(module, init_std):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def _rescale_residual_weights(blocks):
    #deeper layers get smaller weights so the residual sum doesnt blow up
    for layer_id, layer in enumerate(blocks, start=1):
        layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
        layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))


#encoder

class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        #pos embed is fixed, not learned. parameter only so it gets saved in checkpoint
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches ** .5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(lambda m: _init_weights(m, init_std))
        _rescale_residual_weights(self.blocks)

    def forward(self, x, masks=None):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        x = self.patch_embed(x)
        #add position before masking so the kept patches still know where they were
        x = x + self.pos_embed
        if masks is not None:
            x = apply_masks(x, masks)#keep only the patches in the mask
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

#predictor
#smaller vit, takes context tokens + mask tokens at target positions and predicts the target reps

class VisionTransformerPredictor(nn.Module):

    def __init__(self, num_patches, embed_dim=768, predictor_embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))#one learned token for every target patch
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches ** .5))
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=init_std)#has to be before apply() otherwise random init changes
        self.apply(lambda m: _init_weights(m, init_std))
        _rescale_residual_weights(self.predictor_blocks)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)#actual batch size

        #context tokens -> predictor width and add their positions
        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, D = x.shape

        #mask tokens = mask_token + position of the target patch
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        #copy context once for each target block and stick mask tokens at the end
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        #we only need the outputs of the mask tokens, project back to encoder dim
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)
        return x

#model sizes


_LN = partial(nn.LayerNorm, eps=1e-6)

#name: (embed_dim, depth, num_heads, mlp_ratio)
VIT_SIZES = {
    'vit_small': (384, 12, 6, 4),
    'vit_base': (768, 12, 12, 4),
    'vit_large': (1024, 24, 16, 4),
    'vit_huge': (1280, 32, 16, 4),
    'vit_giant': (1408, 40, 16, 48 / 11),
}

#predictor (depth, width) for each model, from the paper
#vit_small is not in the paper and is only 384 wide itself, so predictor is half of that (192)
PREDICTOR = {
    'vit_small': (6, 192),
    'vit_base': (6, 384),
    'vit_large': (12, 384),
    'vit_huge': (12, 384),
    'vit_giant': (16, 384),
}


def make_encoder(model_name, crop_size=224, patch_size=16):
    if model_name not in VIT_SIZES:
        raise ValueError(f'unknown model {model_name!r}, choose from {list(VIT_SIZES)}')
    embed_dim, depth, num_heads, mlp_ratio = VIT_SIZES[model_name]
    return VisionTransformer(img_size=crop_size, patch_size=patch_size, embed_dim=embed_dim,
                             depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=True, norm_layer=_LN)


def build_models(model_name='vit_small', crop_size=224, patch_size=16, pred_depth=12,
                 pred_emb_dim=384, device=torch.device('cpu')):
    encoder = make_encoder(model_name, crop_size=crop_size, patch_size=patch_size)
    predictor = VisionTransformerPredictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,#same heads as encoder
        mlp_ratio=4, qkv_bias=True, norm_layer=_LN)

    #original code does init a second time, linear and layernorm get reset again
    #this undoes the residual rescaling above, conv and mask_token are left as is
    def reinit(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        reinit(m)
    for m in predictor.modules():
        reinit(m)

    encoder.to(device)
    predictor.to(device)
    return encoder, predictor
