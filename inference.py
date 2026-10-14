import argparse
import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import test_transform
from evaluate import features, make_head
from utils import MaskCollator, apply_masks, repeat_interleave_batch
from vit import build_models

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

#accepts image files and/or folders; a folder is replaced by the images inside it (sorted)
def list_images(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            files.append(p)
    return files


#loads the images the same way evaluate.py does: resize, centre crop, normalise.
#crop_size comes from the checkpoint, so the model gets the image size it was trained on.
def load_images(files, crop_size):
    transform = test_transform(crop_size)
    #.convert('RGB'): grayscale or transparent (RGBA) images become normal 3-channel images
    return torch.stack([transform(Image.open(f).convert('RGB')) for f in files])   #(n, 3, crop, crop)


#rebuilds the three networks from a checkpoint written by train.py. the checkpoint stores the
#config of the run, so we know which model size, image size and predictor were used.
def load_models(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    encoder, predictor = build_models(cfg['model'], cfg['crop_size'], cfg['patch_size'],
                                      cfg['predictor']['depth'], cfg['predictor']['width'], device)
    target_encoder, _ = build_models(cfg['model'], cfg['crop_size'], cfg['patch_size'],
                                     cfg['predictor']['depth'], cfg['predictor']['width'], device)
    #the random starting weights of build_models are replaced by the trained ones
    encoder.load_state_dict(ckpt['encoder'])
    predictor.load_state_dict(ckpt['predictor'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    for net in (encoder, predictor, target_encoder):
        net.eval()
    print(f"loaded {cfg['model']}/{cfg['patch_size']} trained on {cfg['dataset']} for {ckpt['epoch']} epochs")
    return encoder, predictor, target_encoder, cfg


#readable class names: STL-10 classes are already names (airplane, bird, ...); ImageNet classes are
#WordNet ids (n01558993), and prepare_data.py writes data/<dataset>/classes.json with their names
def class_names(probe, data_root):
    names_file = os.path.join(data_root, probe['dataset'], 'classes.json')
    if not os.path.exists(names_file):
        return probe['classes']
    with open(names_file) as f:
        names = json.load(f)
    return [f"{c} ({names[c].split(',')[0]})" if c in names else c for c in probe['classes']]


@torch.no_grad()
def classify(probe_path, imgs, files, encoder, data_root, device, topk):
    probe = torch.load(probe_path, map_location='cpu', weights_only=False)
    classes = class_names(probe, data_root)
    #the same kind of head as the best one evaluate.py found: linear or BatchNorm + linear,
    #on the last layer or on the last 4 layers
    head = make_head(probe['dim'], len(classes), probe['batchnorm'])
    head.load_state_dict(probe['state_dict'])
    head = head.to(device).eval()
    last, last4 = features(encoder, imgs, bf16=False)
    logits = head(last if probe['representation'] == 'last layer' else last4)
    probs = logits.softmax(dim=1)                                      #scores -> probabilities
    top_p, top_i = probs.topk(min(topk, len(classes)), dim=1)
    print(f"probe: {probe['representation']}, {'BN + linear' if probe['batchnorm'] else 'linear'}, "
          f"{probe['top1']:.1f}% top-1 on {probe['dataset']}")
    for f, ps, idxs in zip(files, top_p.tolist(), top_i.tolist()):
        print(os.path.basename(f) + ': ' + ', '.join(f'{classes[i]} {p:.1%}' for p, i in zip(ps, idxs)))


@torch.no_grad()
def jepa_demo(imgs, encoder, predictor, target_encoder, cfg, seed=0):
    mk = cfg['mask']
    torch.manual_seed(seed)   #fixed seed: the same masks every time you run it
    collator = MaskCollator(cfg['crop_size'], cfg['patch_size'], tuple(mk['context_scale']),
                            tuple(mk['target_scale']), tuple(mk['target_aspect_ratio']), nenc=1,
                            npred=mk['num_targets'], min_keep=mk['min_keep'], allow_overlap=mk['allow_overlap'])
    #the collator expects (image, label) pairs, so every image gets a dummy label 0
    (batch, _), masks_enc, masks_pred = collator([(img, 0) for img in imgs.cpu()])
    device = imgs.device
    batch = batch.to(device)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    #exactly what train.py does in one step, without updating anything:
    #target = layer-normed target-encoder features of the target blocks (the "answers")
    h = target_encoder(batch)
    h = F.layer_norm(h, (h.size(-1),))
    h = repeat_interleave_batch(apply_masks(h, masks_pred), len(batch), repeat=len(masks_enc))
    #prediction = predictor(context encoder(context patches only), positions of the targets)
    z = predictor(encoder(batch, masks_enc), masks_enc, masks_pred)

    B = len(batch)
    print(f'context: {masks_enc[0].shape[1]} patches, {len(masks_pred)} target blocks of '
          f'{masks_pred[0].shape[1]} patches each')
    for block in range(len(masks_pred)):
        rows = slice(block * B, (block + 1) * B)   #rows are ordered block by block (apply_masks)
        loss = F.smooth_l1_loss(z[rows], h[rows]).item()
        #cosine similarity per predicted patch: 1 = points exactly like the target feature
        cos = F.cosine_similarity(z[rows], h[rows], dim=-1).mean().item()
        print(f'target block {block}: smooth-L1 {loss:.4f}, mean cosine(prediction, target) {cos:.3f}')
    print(f'overall I-JEPA loss: {F.smooth_l1_loss(z, h).item():.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a trained I-JEPA checkpoint on your own images')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+', required=True, help='image files and/or folders')
    parser.add_argument('--probe', help='linear_probe_<dataset>.pth saved by evaluate.py')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--jepa-demo', action='store_true')
    parser.add_argument('--save-embeddings', help='save the embeddings to this file (torch.save)')
    parser.add_argument('--data-root', default='data', help='where classes.json is looked for')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    encoder, predictor, target_encoder, cfg = load_models(args.checkpoint, device)
    files = list_images(args.images)
    imgs = load_images(files, cfg['crop_size']).to(device)

    #embeddings: target encoder, average over all patches -> one vector per image
    last, _ = features(target_encoder, imgs, bf16=False)
    print(f'{len(files)} image(s) -> embeddings of shape {tuple(last.shape)}')
    if args.save_embeddings:
        torch.save({'files': files, 'embeddings': last.cpu()}, args.save_embeddings)
        print(f'saved embeddings to {args.save_embeddings}')

    if args.probe:
        classify(args.probe, imgs, files, target_encoder, args.data_root, device, args.topk)
    if args.jepa_demo:
        jepa_demo(imgs, encoder, predictor, target_encoder, cfg)


if __name__ == '__main__':
    main()
