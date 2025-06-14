import os
import torch
import torch.nn as nn
import numpy as np
import random

from PIL import Image
from torch.utils.data import DataLoader,Dataset,random_split

from torchvision import transforms


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# In[9]:

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器：下采样阶段
        self.conv1 = conv_block(3, 64)

        self.conv2 = conv_block(64, 128)

        self.conv3 = conv_block(128, 256)

        self.conv4 = conv_block(256, 512)

        self.conv5 = conv_block(512, 1024)

        # 解码器：上采样阶段
        self.up_conv1 = up_conv(1024, 512)
        self.conv6 = conv_block(1024, 512)

        self.up_conv2 = up_conv(512, 256)
        self.conv7 = conv_block(512, 256)

        self.up_conv3 = up_conv(256, 128)
        self.conv8 = conv_block(256, 128)

        self.up_conv4 = up_conv(128, 64)
        self.conv9 = conv_block(128, 64)

        self.conv_end = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器阶段
        # In:[B,3,H,W]
        # Out:[B,64,H/2,W/2]
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        # In:[B,64,H/2,W/2]
        # Out:[B,128,H/4,W/4]
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        # In:[B,128,H/4,W/4]
        # Out:[B,256,H/8,W/8]
        x3 = self.conv3(x3)
        x4 = self.max_pool(x3)

        # In:[B,256,H/8,W/8]
        # Out:[B,512,H/16,W/16]
        x4 = self.conv4(x4)
        x5 = self.max_pool(x4)

        # In:[B,512,H/16,W/16]
        # Out:[B,1024,H/16,W/16]
        x5 = self.conv5(x5)

        # 解码器阶段

        # In:[B,1024,H/16,W/16]
        # Out:[B,512,H/8,W/8]
        d5 = self.up_conv1(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.conv6(d5)

        # In:[B,512,H/8,W/8]
        # Out:[B,256,H/4,W/4]
        d6 = self.up_conv2(d5)
        d6 = torch.cat((x3, d6), dim=1)
        d6 = self.conv7(d6)

        # In:[B,256,H/4,W/4]
        # Out:[B,128,H/2,W/2]
        d7 = self.up_conv3(d6)
        d7 = torch.cat((x2, d7), dim=1)
        d7 = self.conv8(d7)

        # In:[B,128,H/2,W/2]
        # Out:[B,64,H,W]
        d8 = self.up_conv4(d7)
        d8 = torch.cat((x1, d8), dim=1)
        d8 = self.conv9(d8)

        # In:[B,64,H,W]
        # Out:[B,1,H,W]
        out = self.conv_end(d8)
        return out