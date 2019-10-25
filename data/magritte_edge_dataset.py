import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset, make_magritte_edge_dataset
from PIL import Image
import torch


class MagritteEdgeDataset(BaseDataset):
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
        self.ABC_paths = make_magritte_edge_dataset(self.dir_AB, opt.max_dataset_size)  # get image paths
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
        B = Image.open(ABC_path['B']).convert('RGB')
        
        # if no SegmentationFake -> all frame is fake
        if os.path.exists(ABC_path['C']): 
            C = Image.open(ABC_path['C']).convert('RGB')
            C_edge = Image.open(ABC_path['C_edge']).convert('RGB')
        else:
            C = Image.new('RGB', A.size, (255, 255, 255))
            C_edge = Image.new('RGB', A.size, (255, 255, 255))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        transform_params_b = get_params(self.opt, B.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params_b, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 2), method=Image.BICUBIC)
        
        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        C_edge = C_transform(C_edge)
        
        if A.shape[1:3] != C.shape[1:3]:
            print('Error A/C shape mismatch A {}, B {}, C {} {}'.format(A.shape, B.shape, C.shape, ABC_path['A']))
        
        C = torch.cat((C, C_edge), 0)        
        

        return {'A': A, 'B': B, 'C': C, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
