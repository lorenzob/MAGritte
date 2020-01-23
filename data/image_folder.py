"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
from random import shuffle
import os
import os.path
import pandas as pd
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_magritte_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_fake       = os.path.join(dir, 'ColorFakeImages')
    dir_real       = os.path.join(dir, 'ColorRealImages')
    dir_label_fake = os.path.join(dir, 'SegmentationFake')

    for root, _, fnames in sorted(os.walk(dir_fake)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_fake = os.path.join(root, fname)
                #path_real = os.path.join(dir_real, fname)
                path_label_fake = os.path.join(dir_label_fake, fname)
                images.append({'A':path_fake, 'B':'', 'C':path_label_fake})
    
    j = 0
    for root, _, fnames in sorted(os.walk(dir_real)):
        for fname in sorted(fnames):
            if is_image_file(fname):                
                path_real = os.path.join(root, fname)
                images[j]['B'] = path_real
                j += 1                
    
    return images[:min(max_dataset_size, len(images))]

def make_magritte_edge_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_fake       = os.path.join(dir, 'ColorFakeImages')
    dir_real       = os.path.join(dir, 'ColorRealImages')
    dir_label_fake = os.path.join(dir, 'SegmentationFake')
    dir_label_fake_egde = os.path.join(dir, 'SegmentationFakeEdge')

    for root, _, fnames in sorted(os.walk(dir_fake)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_fake = os.path.join(root, fname)
                #path_real = os.path.join(dir_real, fname)
                path_label_fake = os.path.join(dir_label_fake, fname)
                path_label_fake_edge = os.path.join(dir_label_fake_egde, fname)
                images.append({'A':path_fake, 'B':'', 'C':path_label_fake, 'C_edge':path_label_fake_edge})
    
    j = 0
    for root, _, fnames in sorted(os.walk(dir_real)):
        for fname in sorted(fnames):
            if is_image_file(fname):                
                path_real = os.path.join(root, fname)
                images[j]['B'] = path_real
                j += 1                
    
    return images[:min(max_dataset_size, len(images))]
    
def make_magritte_sem_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_fake       = os.path.join(dir, 'ColorFakeImages')
    dir_real       = os.path.join(dir, 'ColorRealImages')
    dir_label_fake = os.path.join(dir, 'SegmentationFake')
    dir_sem_real   = os.path.join(dir, 'SegmentationRealClass')
    dir_sem_fake   = os.path.join(dir, 'SegmentationFakeClass')


    for root, _, fnames in sorted(os.walk(dir_fake)):
        shuffle(fnames)
        for fname in fnames:
            if is_image_file(fname):
                path_fake = os.path.join(root, fname)
                path_label_fake = os.path.join(dir_label_fake, fname)
                path_class_fake = os.path.join(dir_sem_fake, fname)
                images.append({'A':path_fake, 'B':'', 'CA_fake':path_label_fake, 'CA_class':path_class_fake, 'CB_class':''})
    
    j = 0
    for root, _, fnames in sorted(os.walk(dir_real)):
        shuffle(fnames)
        for fname in fnames:
            if is_image_file(fname):                
                path_real = os.path.join(root, fname)
                path_sem_real = os.path.join(dir_sem_real, fname)
                images[j]['B'] = path_real
                images[j]['CB_class'] = path_sem_real
                j += 1                
    
    return images[:min(max_dataset_size, len(images))]


def make_magritte_edge_sem_dataset(dir, max_dataset_size=float("inf"), is_train=False):
    dir = Path(dir)
    assert dir.is_dir(), '%s is not a valid directory' % dir
    images = []

    df = pd.read_pickle(os.path.join(dir, "image_db.pkl"))
    fake_df = df[df["is_test"] == (not is_train)][df["is_real"] == 0]
    fake_fnames = fake_df["img_name"].to_list()

    for fname in sorted(fake_fnames):
        path_real = dir.joinpath("ColorRealImages", fname[12:])
        path_label_fake = dir.joinpath("SegmentationFake", fname).with_suffix(".npz")
        path_class_fake = dir.joinpath("SegmentationFakeClass", fname).with_suffix(".npz")
        path_label_fake_edge = dir.joinpath("SegmentationFakeEdge", fname).with_suffix(".npz")
        path_sem_real = dir.joinpath("SegmentationRealClass", fname[12:-4]).with_suffix(".npz")

        images.append({'A': str(dir.joinpath("ColorFakeImages", fname)),
                       'B': str(path_real),
                       'CA_fake': str(path_label_fake),
                       'CA_edge': str(path_label_fake_edge),
                       'CA_class': str(path_class_fake),
                       'CB_class': str(path_sem_real)
                       })
    
    return images[:min(max_dataset_size, len(images))]


def make_ab_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    dir_fake       = os.path.join(dir, 'ColorFakeImages')
    dir_real       = os.path.join(dir, 'ColorRealImages')

    for root, _, fnames in sorted(os.walk(dir_fake)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path_fake = os.path.join(root, fname)
                path_real = os.path.join(dir_real, fname)
                images.append({'A':path_fake, 'B':path_real})
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
