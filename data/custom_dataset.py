import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
import tifffile
import torchvision.transforms as transforms

class CustomDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=4, output_nc=3)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # 使用tifffile加载TIFF影像
        A_img = tifffile.imread(A_path)
        B_img = tifffile.imread(B_path)

        # 确保数据类型为np.float32
        A_img = A_img.astype(np.float32)
        B_img = B_img.astype(np.float32)

        # 打印图像的形状信息，用于调试
        #print(f"A_img shape before transpose: {A_img.shape}")
        #print(f"B_img shape before transpose: {B_img.shape}")
        if A_img.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels, but got {A_img.shape[-1]} channels for A_img")
        if B_img.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels, but got {B_img.shape[-1]} channels for B_img")
        # 检查通道数
        if A_img.ndim == 3:
            # 转换为CHW格式
            A_img = np.transpose(A_img, (2, 0, 1))
        else:
            # 处理单通道或其他情况
            A_img = np.expand_dims(A_img, axis=0)

        if B_img.ndim == 3:
            B_img = np.transpose(B_img, (2, 0, 1))
        else:
            B_img = np.expand_dims(B_img, axis=0)

        # 再次打印图像的形状信息，用于调试
        #print(f"A_img shape after transpose: {A_img.shape}")
        #print(f"B_img shape after transpose: {B_img.shape}")

        # 确保通道数不超过4
        if A_img.shape[0] > 4:
            print(f"Warning: A_img has {A_img.shape[0]} channels, truncating to 4.")
            A_img = A_img[:4, :, :]
        if B_img.shape[0] > 4:
            print(f"Warning: B_img has {B_img.shape[0]} channels, truncating to 4.")
            B_img = B_img[:4, :, :]

        # 将CHW格式转换为HWC格式
        A_img = np.transpose(A_img, (1, 2, 0))
        B_img = np.transpose(B_img, (1, 2, 0))

        # 转换为PIL图像
        A_pil = transforms.functional.to_pil_image(A_img)
        B_pil = transforms.functional.to_pil_image(B_img)

        # 应用转换
        A = self.transform(A_pil)
        B = self.transform(B_pil)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)