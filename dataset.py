import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#settings for each dataset:
#crop_size  = size of the image that goes into the model (224 for ImageNet, 96 for STL-10)
#patch_size = size of one patch, so 224/16 = 14x14 patches, 96/8 = 12x12 patches
#splits     = which split is used for (pre-training, evaluation train, evaluation test)

DATASETS = {
    'imagenet100': dict(crop_size=224, patch_size=16, splits=('train', 'train', 'val')),
    'imagenet1k': dict(crop_size=224, patch_size=16, splits=('train', 'train', 'val')),
    'stl10': dict(crop_size=96, patch_size=8, splits=('train+unlabeled', 'train', 'test')),
}

class STL10(torch.utils.data.Dataset):
    #which files belong to each split: (images file, labels file); unlabeled has no labels file
    FILES = {'train': [('train_X.bin', 'train_y.bin')],
             'test': [('test_X.bin', 'test_y.bin')],
             'train+unlabeled': [('train_X.bin', 'train_y.bin'), ('unlabeled_X.bin', None)]}

    def __init__(self, root, split, transform=None):
        folder = os.path.join(root, 'stl10_binary')
        self.transform = transform
        self.files, labels = [], []
        for x_file, y_file in self.FILES[split]:
            #every image is 3 x 96 x 96 bytes, so the file size tells us how many images it has
            n = os.path.getsize(os.path.join(folder, x_file)) // (3 * 96 * 96)
            self.files.append((os.path.join(folder, x_file), n))
            #labels are stored as 1..10, we want 0..9; unlabeled images get -1 (never used)
            if y_file:
                labels.append(np.fromfile(os.path.join(folder, y_file), dtype=np.uint8).astype(np.int64) - 1)
            else:
                labels.append(np.full(n, -1, dtype=np.int64))
        self.labels = np.concatenate(labels)
        with open(os.path.join(folder, 'class_names.txt')) as f:
            self.classes = [c.strip() for c in f if c.strip()]   #airplane, bird, car, ...
        self.arrays = None   #the files are opened on first use, inside each worker process

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = [np.memmap(p, dtype=np.uint8, mode='r', shape=(n, 3, 96, 96)) for p, n in self.files]
        #find which file the index falls in (train+unlabeled = train file first, then unlabeled file)
        i = index
        for arr in self.arrays:
            if i < len(arr):
                break
            i -= len(arr)
        #the file stores each image as (3, width, height); PIL wants (height, width, 3)
        img = Image.fromarray(np.ascontiguousarray(arr[i].transpose(2, 1, 0)))
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[index])


#returns the dataset for one split: 'pretrain', 'eval_train' or 'eval_test'
#each item is (image, label); the label is ignored during pre-training
def build_dataset(name,root,split,transform):
    if name not in DATASETS:
        raise ValueError(f'unknown dataset {name!r}, choose from {list(DATASETS)}')
    #translate 'pretrain' / 'eval_train' / 'eval_test' into the real split name, e.g. 'val'
    real_split = dict(zip(('pretrain', 'eval_train', 'eval_test'), DATASETS[name]['splits']))[split]
    path = os.path.join(root, name)   #e.g. data/imagenet100
    if name == 'stl10':
        if not os.path.isdir(os.path.join(path, 'stl10_binary')):
            raise FileNotFoundError(f'{path}/stl10_binary not found: run python prepare_data.py --dataset stl10')
        return STL10(path, real_split, transform)
    #ImageNet: one sub-folder per class, ImageFolder gives each folder a label 0, 1, 2, ... (sorted names)
    return torchvision.datasets.ImageFolder(os.path.join(path, real_split), transform=transform)


def make_pretrain_loader(
    dataset,
    collator,
    batch_size,
    num_workers,
    persistent_workers=True
):
    #which dataset items are selected and the order they are provided.
    #we shuffle because seeing the images in order (all dogs, then all cats,) would be bad for training.
    #DistributedSampler with one "GPU" just shuffles, with seed 0 + epoch (same as the original code),
    #so the order is reproducible and stays the same after resuming. train.py calls sampler.set_epoch(epoch)
    #every epoch to get a new order (RandomSampler has no set_epoch, that's why it is not used).
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=1, rank=0)
    #receives the examples from the sampler and turns them into batches.
    #the collator (MaskCollator in utils.py) also samples the context and target masks for each batch, so one batch is
    #([images (128,3,224,224), labels (128)], context masks, target masks)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,          #drops the last batch if it has fewer than batch_size images
        pin_memory=True,      
        num_workers=num_workers, 
        persistent_workers=persistent_workers and num_workers > 0)
    return loader, sampler


#loader for evaluation: plain (image, label) batches, no masks.
#shuffle=False when extracting features, so the features stay in the same order as the labels;
#shuffle=True when training the linear probe.
def make_eval_loader(dataset,batch_size,shuffle,num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)       #every image counts in evaluation


#transforms during pre-training.
#I-JEPA only uses a random crop: no flip, colour jitter or blur (the paper's point is that it does
#not need hand-made augmentations), so those options of the original code are not here.
def pretrain_transform(crop_size,crop_scale=(0.3, 1.0)):
    #this randomly crops 30-100% of the image and then resizes it to crop_size (224), this gives
    #different views of the images. like if we take an img with a dog and lots of grass, the crop may
    #sometimes contain the dog and sometimes only grass.
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=crop_scale),
        T.ToTensor(),                                #converts the image to a (3, 224, 224) tensor in [0, 1]
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


#transforms for training the linear probe: random crop + horizontal flip (the VISSL / MAE recipe the
#paper uses). only the small linear head is trained here, the encoder stays frozen.
def probe_train_transform(crop_size):
    return T.Compose([
        T.RandomResizedCrop(crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


#now transforms during eval: no randomness, every image is processed the same way.
#resize the shorter side to crop_size / 0.875 (256 for 224) and cut out the centre crop_size x crop_size.
def test_transform(crop_size):
    return T.Compose([
        T.Resize(int(round(crop_size / 0.875)), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
# here BICUBIC is an interpolation method used to estimate pixel values when resizing an image.
# it uses the neighbouring pixels (4x4) to estimate new pixels, often producing smooth results.
