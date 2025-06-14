import os
import torch
import torch.nn as nn
import numpy as np
import random

from PIL import Image
from torch.utils.data import DataLoader,Dataset,random_split

from torchvision import transforms

class ImgData(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.image_files[index]))
        mask = Image.open(os.path.join(self.mask_path, self.image_files[index]))

        if self.transform:
            img, mask = self.transform(img), self.transform(mask)

        return img, mask*255.0