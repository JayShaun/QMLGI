#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> data
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2022/7/18 19:52
@Desc   ：
=================================================='''
from torch.utils.data import DataLoader,Dataset
import struct
import numpy as np
import cv2
import os


class mnistDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(img_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28 * 28)
        self.labels = labels
        self.images = images
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        if self.transform is not None:
            return self.transform(self.images[i]), self.labels[i]
        return self.images[i], self.labels[i]


class bucketsDataset(Dataset):
    def __init__(self, buckets_path, label_path, transform=None, flag='train'):
        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        buckets = np.load(buckets_path)
        self.labels = labels
        self.buckets = buckets
        self.transform = transform
        if not flag=='all':
            self.labels = labels[0:int(flag)]
            self.buckets = buckets[0:int(flag)]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        if self.transform is not None:
            return self.transform(self.buckets[i]), self.labels[i]
        return self.buckets[i], self.labels[i]

class planeDataset(Dataset):
    def __init__(self, img_path, buckets_path, transform1=None, transform2=None):
        self.imgs_path = [os.path.join(img_path, name) for name in os.listdir(img_path)]
        self.buckets = np.load(buckets_path)
        print(self.buckets.min())
        self.transform1 = transform1
        self.transform2 = transform2
        print(len(self.buckets), len(self.imgs_path))
    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self, i):
        img = cv2.imread(self.imgs_path[i], 0)
        img = np.reshape(img, [1, 64, 64])
        if self.transform1 is not None:
            return self.transform1(img), self.transform2(self.buckets[i])
        return img, self.buckets[i]

class bucketsDataset1(Dataset):
    def __init__(self, buckets_path):

        buckets = np.load(buckets_path)
        self.buckets = buckets

    def __len__(self):
        return len(self.buckets)
    def __getitem__(self, i):
        return self.buckets[i]



