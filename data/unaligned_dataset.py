import os

import rasterio
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
import random
import torchvision.transforms as transforms

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 4))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 3))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = self.load_image(A_path)
        B_img = self.load_image(B_path)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def load_image(self, path):
        if path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(path) as src:
                img = src.read()
                img = np.transpose(img, (1, 2, 0))
        else:
            img = Image.open(path).convert('RGB')
            img = np.array(img)
        return img

    def __len__(self):
        return max(self.A_size, self.B_size)