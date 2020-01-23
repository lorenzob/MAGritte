import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_np
import torchvision.transforms as transforms
from data.image_folder import make_dataset, make_magritte_edge_sem_dataset
from PIL import Image
import torch
import numpy as np


class MagritteEdgeSemDataset(BaseDataset):
    """A dataset class for fake, real and mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image triplets in the form of {A,B,C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.ABC_paths = make_magritte_edge_sem_dataset(opt.dataroot, opt.max_dataset_size, opt.isTrain)  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABC_path = self.ABC_paths[index]

        # Images
        A = Image.open(ABC_path['A']).convert('RGB')
        B = Image.open(ABC_path['B']).convert('RGB')

        # Markup
        CA_class_raw = np.load(ABC_path['CA_class'])["arr_0"]
        CB_class_raw = np.load(ABC_path['CB_class'])["arr_0"]
        assert CA_class_raw.shape[:2] == CB_class_raw.shape[:2], "Markup shape mismatch"
        CA_class = np.zeros(list(CA_class_raw.shape[:2]) + [10], dtype=np.uint8)
        CB_class = CA_class.copy()
        for j in range(10):  # Convert markup
            CA_class[:, :, j] = (CA_class_raw == j).any(axis=-1)
            CB_class[:, :, j] = (CB_class_raw == j).any(axis=-1)

        # Fake mask
        CA_fake_raw = np.load(ABC_path['CA_fake'])["arr_0"]
        CA_edge_raw = np.load(ABC_path['CA_edge'])["arr_0"]
        CA_fake = np.reshape(CA_fake_raw, CA_fake_raw.shape[:2]).astype(bool).astype(np.uint8) * 255
        CA_edge = np.reshape(CA_edge_raw, CA_fake_raw.shape[:2]).astype(bool).astype(np.uint8) * 255
        assert CA_fake_raw.shape[:2] == CA_edge_raw.shape[:2], "Fake mask shape mismatch"

        # Get transform
        transform_params_a = get_params(self.opt, A.size)
        transform_params_b = get_params(self.opt, B.size)
        A_transform = get_transform(self.opt, transform_params_a, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params_b, grayscale=(self.input_nc == 1), method=Image.NEAREST)

        CA_class_transform = get_transform_np(self.opt, chanels=CA_class.shape[2], params=transform_params_a)
        CB_class_transform = get_transform_np(self.opt, chanels=CB_class.shape[2], params=transform_params_b)

        CA_fake_transform = get_transform(self.opt, transform_params_a, grayscale=True)

        # Apply transform
        A = A_transform(A)
        B = B_transform(B)
        CA_class = CA_class_transform(CA_class)
        CB_class = CB_class_transform(CB_class)
        CA_fake = CA_fake_transform(Image.fromarray(CA_fake))
        CA_edge = CA_fake_transform(Image.fromarray(CA_edge))
        
        # fake image semantic labeling + fake image fake regions 
        CA = torch.cat(
            (CA_class, CA_fake, CA_edge),
            0
        )
        # real image semantic labeling + real image fake regions 
        CB = torch.cat(
            (CB_class, B.new_full((1, B.shape[2], B.shape[2]), -1.0), B.new_full((1, B.shape[2], B.shape[2]), -1.0)),
            0
        )

        return {'A': A, 'B': B, 'CA': CA, 'CB' : CB, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
