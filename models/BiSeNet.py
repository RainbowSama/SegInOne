import os
import torch
import torch.nn as nn
import numpy as np
import random

import torchvision
from PIL import Image
from torch.utils.data import DataLoader,Dataset,random_split

from torchvision import transforms

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AttentionRefineModule(nn.Module):
    def __init__(self, in_ch,out_ch, *args,**kwargs):
        super(AttentionRefineModule, self).__init__()
        self.conv=ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_att=nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_att=nn.BatchNorm2d(out_ch)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        feat=self.conv(x)
        att=nn.functional.avg_pool2d(feat,feat.size()[2:])
        att=self.conv_att(att)
        att=self.bn_att(att)
        att=self.sigmoid(att)
        out=torch.mul(x,att)
        return out

class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch,out_ch, *args,**kwargs):
        super(FeatureFusionModule, self).__init__()

        self.conv=ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.global_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, sp,cp):
        x=torch.cat([sp,cp],dim=1)
        assert self.in_ch==x.size(1),'in_channels of ConvBNReLU must be {}'.format(x.size(1))

        feat=self.conv(x)

        x=self.global_pool(feat)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.sigmoid(x)
        mul_feat=torch.mul(x,feat)
        feat=feat+mul_feat
        return feat

class SpatialPath(nn.Module):
    def __init__(self,*args,**kwargs):
        super(SpatialPath,self).__init__()
        self.conv1=ConvBNReLU(3,64,kernel_size=7,stride=2,padding=3)
        self.conv2=ConvBNReLU(64,64,kernel_size=3,stride=2,padding=1)
        self.conv3=ConvBNReLU(64,64,kernel_size=3,stride=2,padding=1)
        self.conv_out=ConvBNReLU(64,128,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        feat=self.conv1(x)
        feat=self.conv2(feat)
        feat=self.conv3(feat)
        feat=self.conv_out(feat)
        return feat

class ContextPath(nn.Module):
    def __init__(self,*args,**kwargs):
        super(ContextPath,self).__init__()
        self.resnet=torchvision.models.resnet18(weights=None)
        self.sollow=nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool)
        self.layer1=self.resnet.layer1
        self.layer2=self.resnet.layer2
        self.layer3=self.resnet.layer3
        self.layer4=self.resnet.layer4

        self.arm16=AttentionRefineModule(256,128)
        self.arm32=AttentionRefineModule(512,128)
        self.conv_head32=ConvBNReLU(128,128,kernel_size=3,stride=1,padding=1)
        self.conv_head16=ConvBNReLU(128,128,kernel_size=3,stride=1,padding=1)
        self.conv_avg=ConvBNReLU(512,128,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        H0,W0=x.size()[2:]
        feat8=self.layer2(self.layer1(self.sollow(x)))
        feat16=self.layer3(feat8)
        feat32=self.layer4(feat16)
        H8,W8=feat8.size()[2:]
        H16,W16=feat16.size()[2:]
        H32,W32=feat32.size()[2:]

        avg=nn.functional.avg_pool2d(feat32,feat32.size()[2:])
        avg=self.conv_avg(avg)
        avg_up=nn.functional.interpolate(avg,(H32,W32),mode='nearest')

        feat32_arm=self.arm32(avg_up)
        feat32_sum=feat32_arm+avg_up
        feat32_up=nn.functional.interpolate(feat32_sum,(H16,W16),mode='nearest')
        feat32_up=self.conv_head32(feat32_up)

        feat16_arm=self.arm16(feat16)
        feat16_sum=feat16_arm+feat32_up
        feat16_up=nn.functional.interpolate(feat16_sum,(H8,W8),mode='nearest')
        feat16_up=self.conv_head16(feat16_up)

        return feat16_up,feat32_up


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class BiSeNet(nn.Module):
    def __init__(self, num_classes=256, pretrained=True):
        super(BiSeNet, self).__init__()
        self.spatial_path=SpatialPath()
        self.context_path=ContextPath()

        self.ffm=FeatureFusionModule(256,256)

        self.out_conv=BiSeNetOutput(256,256,num_classes)

    def forward(self, x):
        sp=self.spatial_path(x)
        cp16_up,cp32_up=self.context_path(sp)
        ff=self.ffm(sp,cp16_up)
        out=self.out_conv(ff)
        return out