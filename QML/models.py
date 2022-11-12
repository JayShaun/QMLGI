#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> models
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2022/7/18 15:52
@Desc   ：
=================================================='''
import torch.nn as nn
import torch
import torch.nn.modules as nn
import torch.nn.functional as F
class bigNet(nn.Module):
    def __init__(self, in_dim = 64, out_dim = 10):
        super(bigNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
        )
        self.clc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        fea = self.net(x)
        y = self.clc(fea)
        return fea, y

class QNN_like(nn.Module):
    def __init__(self, in_dim = 16, embed_dim = 14, out_dim = 10):
        super(QNN_like, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim, out_features=out_dim),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):

        return x, self.net(x)

class QNN_like_2(nn.Module):
    def __init__(self, in_dim = 16, embed_dim = 16, embed_dim_2 = 16, out_dim = 10):
        super(QNN_like_2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim_2),
        )
        self.clc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=embed_dim_2, out_features=out_dim),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        fea = self.net(x)
        y = self.clc(fea)
        return fea, y

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out，out_2
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 
        :param out:
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class UNet5(nn.Module):
    def __init__(self, in_ch = 1):
        super(UNet5, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(in_ch,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],in_ch,3,1,1),
            #nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

class UNet3(nn.Module):
    def __init__(self, in_ch = 1):
        super(UNet3, self).__init__()
        out_channels=[2**(i+6) for i in range(3)] #[64, 128, 256]
        #
        self.d1=DownsampleLayer(in_ch,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        #self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        #self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #
        #self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        #self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[1],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],in_ch,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        #out_3,out3=self.d3(out2)
        #out_4,out4=self.d4(out3)
        #out5=self.u1(out4,out_4)
        #out6=self.u2(out5,out_3)
        out7=self.u3(out2,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out





class uunet(nn.Module):
    def __init__(self, in_dim=64, in_ch=1):
        super(uunet, self).__init__()
        self.fc = nn.Linear(in_dim, 4096)
        self.unet = UNet3(in_ch=in_ch)
    def forward(self, x):
        y = self.fc(x)
        b, d = y.shape
        y = y.view(b, 1, 64, 64)
        y = self.unet(y)
        return y

class CMLnet(nn.Module):
    def __init__(self, in_dim=64,):
        super(qqnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1024), nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(64),
            nn.Conv2d(32, 16, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv2d(8, 4, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.Conv2d(4, 1, (3, 3), 1, 1),
        )
    def forward(self, x):
        x = self.fc(x)
        b, d = x.shape
        y = x.view(b, 1, 32, 32)
        y = self.net(y)
        return y

class CMLnet_patch(nn.Module):
    def __init__(self, in_dim=64,):
        super(qqnet_patch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim//4, in_dim//4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_dim//4, 1024),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 16, (3, 3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 16, (3, 3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 8, (3, 3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 4, (3, 3), 1, 1),
            nn.LeakyReLU(True),
            nn.Conv2d(4, 1, (3, 3), 1, 1),
        )
    def forward(self, x):
        b, l = x.shape
        #print(b, l)
        y1 = self.fc(x[:,0:l//4])
        y2 = self.fc(x[:,l//4:l//2])
        y3 = self.fc(x[:, l//2:int(3*l)//4])
        y4 = self.fc(x[:, int(3*l)//4::])
        y = torch.cat([y1, y2, y3, y4])
        y = y.view(b, 1, 64, 64)
        y = self.net(y)
        return y





