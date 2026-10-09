#here dataset has 3 imp responsibilities
#1.load images from disk.
#2. Transform images into a suitable numerical format.
#3.Deliver batches of iumages to the traiuning code

import logging
import os#it builds file and folder paths
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFilter
logger = logging.getLogger("ijepa")

#these values are for the 3 channels rgb
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)


        return img.filter(ImageFilter.GaussianBlur(radius=radius.item()))
def make_transforms(
        crop_size = 224,
        crop_scale = (0.3,1.0),
        color_jitter=1.0,
        horizontal_flip = False,
        color_distortion = False,
        gaussian_blur=False,
        normalization = (IMAGENET_MEAN,IMAGENET_STD),
):
    #this randomly crops the image and then resizes it to 224, this helps in giving different views of imgs.
    #like if we take an img with dog and lots of grass,the crop may contain sometimes a dog or grass.
    transf_list = [transforms.RandomResizedCrop( crop_size,scale=crop_scale)]
    if horizontal_flip:
        transf_list+=[transforms.RandomHorizontalFlip()]
    if color_distortion:
        s = color_jitter
        transf_list += [transforms.Compose([transforms.RandomApply(
                    [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],p=0.8),#brightness,contrast,sat,hue
                    transforms.RandomGrayscale(p=0.2)])]
    if gaussian_blur:
        transf_list += [GaussianBlur(p=0.5)]
    transf_list+=[transforms.ToTensor(),#converts image to (3,224,224)
                  transforms.Normalize(normalization[0],normalization[1])]
    return transforms.Compose(transf_list)


def make_pretrain_loader(
    root_path,
    image_folder,
    transform,
    collator,
    batch_size,
    num_workers=10,
    pin_mem=True,
    drop_last=True
):
    data_path = os.path.join( root_path, image_folder, "train/")
    logger.info(f"data-path {data_path}")
    dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transform)
    # which dataset items are selected and the order they are provided
    #this is desired cause during training seeing images in the orderr can be undesirable.
    sampler = torch.utils.data.RandomSampler(dataset)
    # recieves the examples from sampler and turns them into batches
    # we are taking 128 images in a batch so (128,3,224,224)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,#drops last batch if imgs < 128
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True
    )

    return dataset, loader, sampler

#now transforms during eval

def make_eval_transform(crop_size = 224):
    resize = int(round(crop_size * 256 / 224))
    return transforms.Compose([transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
# here BICUBIC is an interpolation method used to estimate pixel values when resizing an image
#it uses neighbouring pizels to estimate new pixels, often producing smooth  results

def make_eval_loader(
    root_path,
    image_folder,
    split,
    crop_size=224,
    batch_size=256,
    num_workers=10,
    pin_mem=True
):
    data_path = os.path.join(root_path, image_folder, split)
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=make_eval_transform(crop_size))
    loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False,)
    return dataset, loader
