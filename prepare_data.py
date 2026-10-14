import argparse
import csv
import glob
import json
import os
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
HF = 'https://huggingface.co/datasets/clane9/imagenet-100/resolve/main/'
CMC = 'https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt'


def fetch(url, dst):
    if not os.path.exists(dst):
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(url, dst + '.part')
                os.replace(dst + '.part', dst)
                print('downloaded', os.path.basename(dst), flush=True)
                return
            except OSError:
                if attempt == 4:
                    raise


def prepare_imagenet100():
    import pyarrow.parquet as pq
    raw, out = os.path.join(DATA, 'raw', 'imagenet100'), os.path.join(DATA, 'imagenet100')
    os.makedirs(raw, exist_ok=True)
    shards = [f'data/train-{i:05d}-of-00017.parquet' for i in range(17)] + ['data/validation-00000-of-00001.parquet']
    fetch(CMC, os.path.join(raw, 'imagenet100.txt'))
    fetch(HF + 'README.md', os.path.join(raw, 'README.md'))
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda s: fetch(HF + s, os.path.join(raw, os.path.basename(s))), shards))

    wnids = [line.strip() for line in open(os.path.join(raw, 'imagenet100.txt')) if line.strip()]
    card = open(os.path.join(raw, 'README.md'), encoding='utf-8').read()
    names = {int(i): n.strip() for i, n in re.findall(r"'(\d+)': (.+)", card)}
    assert len(wnids) == 100 and len(names) == 100
    for split, pattern in (('train', 'train-*.parquet'), ('val', 'validation-*.parquet')):
        for w in wnids:
            os.makedirs(os.path.join(out, split, w), exist_ok=True)
        n = 0
        for shard in sorted(glob.glob(os.path.join(raw, pattern))):
            for rb in pq.ParquetFile(shard).iter_batches(batch_size=512, columns=['image', 'label']):
                for img, label in zip(rb.column('image').to_pylist(), rb.column('label').to_pylist()):
                    ext = '.png' if img['bytes'][1:4] == b'PNG' else '.jpg'
                    path = os.path.join(out, split, wnids[label], f'{split}_{n:06d}{ext}')
                    if not os.path.exists(path):
                        with open(path, 'wb') as f:
                            f.write(img['bytes'])   # written as-is, no re-encoding
                    n += 1
        print(f'{split}: {n} images')
    with open(os.path.join(out, 'classes.json'), 'w') as f:
        json.dump({w: names[i] for i, w in enumerate(wnids)}, f, indent=1)
    print(f'done. {raw} can be deleted now.')


def prepare_stl10(src=None):
    out = os.path.join(DATA, 'stl10')
    target = os.path.join(out, 'stl10_binary')
    if not os.path.isdir(target):
        if src:
            shutil.copytree(src, target)
        else:
            import torchvision
            torchvision.datasets.STL10(root=out, split='train', download=True)   # downloads and extracts
    from dataset import STL10
    for split in ('train+unlabeled', 'train', 'test'):
        print(f'stl10 {split}: {len(STL10(out, split))} images')


def organize_imagenet_val(val_dir, labels_path):
    labels = {}
    with open(labels_path) as f:
        if labels_path.lower().endswith('.csv'):
            for row in csv.DictReader(f):                       # Kaggle: ImageId,PredictionString
                labels[row['ImageId']] = row['PredictionString'].split()[0]
        else:
            for line in f:
                if line.strip():
                    name, wnid = line.split()[:2]
                    labels[os.path.splitext(os.path.basename(name))[0]] = wnid
    moved = 0
    for name in sorted(os.listdir(val_dir)):
        wnid = labels.get(os.path.splitext(name)[0])
        if wnid and os.path.isfile(os.path.join(val_dir, name)):
            os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)
            os.replace(os.path.join(val_dir, name), os.path.join(val_dir, wnid, name))
            moved += 1
    folders = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    print(f'moved {moved} images into {len(folders)} class folders')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['stl10', 'imagenet100', 'imagenet1k'])
    parser.add_argument('--src', help='stl10: an existing stl10_binary folder to copy')
    parser.add_argument('--val-dir', help='imagenet1k: flat folder with the 50,000 val images')
    parser.add_argument('--val-labels', help='imagenet1k: LOC_val_solution.csv or "<file> <wnid>" lines')
    args = parser.parse_args()
    if args.dataset == 'imagenet100':
        prepare_imagenet100()
    elif args.dataset == 'stl10':
        prepare_stl10(args.src)
    elif args.val_dir and args.val_labels:
        organize_imagenet_val(args.val_dir, args.val_labels)
    else:
        parser.error('imagenet1k needs --val-dir and --val-labels (see the top of this file)')
