# vision_transformer

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import trunc_normal_, apply_masks, repeat_interleave_batch



# Positional Encoding
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega                     # [D/2]
    pos = pos.reshape(-1)                           # [M]
    out = np.einsum('m,d->md', pos, omega)          # [M, D/2] outer product
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)   # [M, D]


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)              # w goes first (as in the original)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])   # [H*W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])   # [H*W, D/2]
    return np.concatenate([emb_h, emb_w], axis=1)                         # [H*W, D]

## Transformer Block 

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
        self.scale = head_dim ** -0.5                      # 1/sqrt(d_head)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # one matmul gives q, k and v
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3C] -> [B, N, 3, heads, C/heads] -> [3, B, heads, N, C/heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale      # [B, heads, N, N] token similarities
        attn = attn.softmax(dim=-1)                        # each row sums to 1
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)    # weighted sum of values, heads merged
        return self.proj(x)


class Block(nn.Module):
    

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, norm_layer):
        super().__init__()
        # Creation order (norm1, attn, norm2, mlp) matches the original -> same random init.
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):  # to convert image into vectors

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # [B, D, 14, 14] -> flatten(2) -> [B, D, 196] -> transpose -> [B, 196, D]
        return self.proj(x).flatten(2).transpose(1, 2)

# weight initialisation 
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

    for layer_id, layer in enumerate(blocks, start=1):
        layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
        layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))


# The encoder (used twice: context encoder and target encoder)

class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # Registered as a Parameter (so it appears in checkpoints) but frozen: requires_grad=False
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
        x = self.patch_embed(x)          # [B, N, D]
        x = x + self.pos_embed           # position info is added BEFORE masking, so every kept
                                         # token still knows where it came from
        if masks is not None:
            x = apply_masks(x, masks)    # [len(masks)*B, K, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

# The predictor (a narrow ViT that predicts target-block representations)


class VisionTransformerPredictor(nn.Module):

    def __init__(self, num_patches, embed_dim=768, predictor_embed_dim=384, depth=6,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
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
        trunc_normal_(self.mask_token, std=init_std)     # must happen BEFORE apply() (RNG order)
        self.apply(lambda m: _init_weights(m, init_std))
        _rescale_residual_weights(self.predictor_blocks)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)                        # true batch size

        # 1. context tokens: [nenc*B, K_enc, D] -> [nenc*B, K_enc, Dp], plus their positions
        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)          # [B, N, Dp]
        x += apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, D = x.shape

        # 2. mask tokens for the target positions: [npred*nenc*B, K_pred, Dp]
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)                          # [npred*B, K_pred, Dp]
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        # 3. one copy of the context per target block, then concatenate along the token axis
        x = x.repeat(len(masks), 1, 1)                                   # [npred*nenc*B, K_enc, Dp]
        x = torch.cat([x, pred_tokens], dim=1)                           # [.., K_enc + K_pred, Dp]
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # 4. predictions for the mask tokens only, back to encoder width
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)                                       # [npred*nenc*B, K_pred, D]
        return x

# Model sizes and construction


_LN = partial(nn.LayerNorm, eps=1e-6)

# name: (embed_dim, depth, num_heads, mlp_ratio)
VIT_SIZES = {
    'vit_small': (384, 12, 6, 4),
    'vit_base': (768, 12, 12, 4),
    'vit_large': (1024, 24, 16, 4),
}


def make_encoder(model_name, crop_size=224, patch_size=16):
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
        num_heads=encoder.num_heads,    # predictor uses the encoder's head count (6 for ViT-S)
        mlp_ratio=4, qkv_bias=True, norm_layer=_LN)

    # Second initialisation pass, exactly as in the original init_model(): every Linear gets
    # trunc_normal(std=0.02) again and every LayerNorm is reset. This is applied in modules()
    # order (parent before children) and OVERWRITES the residual rescaling done above.
    # Conv2d (patch embedding) and mask_token are not touched here.
    
    ## Reinit
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
