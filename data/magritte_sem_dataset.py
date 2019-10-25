import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset, make_magritte_sem_dataset
from PIL import Image, ImageFilter
import torch


class MagritteSemDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABC_paths = make_magritte_sem_dataset(self.dir_AB, opt.max_dataset_size)  # get image paths
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
        A = Image.open(ABC_path['A']).convert('RGB')
        #A = A.filter(ImageFilter.GaussianBlur(6))
        B = Image.open(ABC_path['B']).convert('RGB')
        #B = B.filter(ImageFilter.GaussianBlur(6))
        CA_class = Image.open(ABC_path['CA_class']).convert('RGB')
        CB_class = Image.open(ABC_path['CB_class']).convert('RGB')
        
        # if no SegmentationFake -> all frame is fake
        if os.path.exists(ABC_path['CA_fake']): 
            CA_fake = Image.open(ABC_path['CA_fake']).convert('RGB')
        else:
            CA_fake = Image.new('RGB', A.size, (255, 255, 255))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        CA_fake_transform = get_transform(self.opt, transform_params, grayscale=True)
        CB_class_transform = get_transform(self.opt, transform_params, grayscale=False)

        A = A_transform(A)
        B = B_transform(B)
        CA_fake  = CA_fake_transform(CA_fake)
        CA_class = CB_class_transform(CA_class)
        CB_class = CB_class_transform(CB_class)

        # fake image semantic labeling + fake image fake regions 
        CA = torch.cat((CA_class, CA_fake), 0)
        # real image semantic labeling + real image fake regions 
        CB = torch.cat((CB_class, B.new_full((1, B.shape[2], B.shape[2]), -1.0)), 0)

        return {'A': A, 'B': B, 'CA': CA, 'CB' : CB, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
