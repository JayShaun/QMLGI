#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> res_model
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2023/5/6 21:09
@Desc   ：
=================================================='''
import torch.nn as nn
import torch.nn.functional as F
import torch
class ResidualBlock(nn.Module):
    def __init__(self,in_channles, num_channles, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channles,num_channles,kernel_size=3,stride=strides,padding=1,)
        self.conv2 = nn.Conv2d(
            num_channles, num_channles, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(num_channles)
        self.bn2 = nn.BatchNorm2d(num_channles)
    def forward(self,x):
        y= F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        y+=x
        return F.relu(y)


class Res_based_net(nn.Module): # Number of parameter:  67824163
    def __init__(self, in_dim=64,):
        super(Res_based_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.Dropout(),
            nn.Linear(4096, int(4*4096)),
            nn.Dropout(),
        )
        self.base = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(),
        )
        self.net1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.net2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.UpsamplingBilinear2d(128),
        )
        self.net3 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.UpsamplingBilinear2d(64),
            nn.UpsamplingBilinear2d(128),
        )
        self.net4 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.UpsamplingBilinear2d(32),
            nn.UpsamplingBilinear2d(64),
            nn.UpsamplingBilinear2d(128),
        )
        self.output = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.MaxPool2d(3,2,1),

            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Conv2d(64, 32, (3, 3), 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Conv2d(32, 1, (3, 3), 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
    def forward(self, x):
        x = self.fc(x)
        b, d = x.shape
        x = x.view(b, 1, 128, 128)
        x = self.base(x)
        y1 = self.net1(x)
        y2 = self.net2(x)
        y3 = self.net3(x)
        y4 = self.net4(x)
        y = torch.cat([y1, y2, y3, y4], 1)
        y = self.output(y)
        return y

if __name__ == '__main__':
    net = stnet(in_dim=64)
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: ", total)