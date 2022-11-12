#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> train_imagingcv2.imwrite保存的图片无法打开
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2022/8/29 9:17
@Desc   ：
=================================================='''

import torch
from models import uunet, qqnet, ggnet, qqnet_patch
from data import planeDataset
import os, sys
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from utils import Logger, compute_ssim, compute_psnr


net = 'qqnet' #['uunet', 'qqnet', 'ggnet', 'qqnet_patch']
in_dim = 128
batch_size = 16
batch_size_test = 1
log_interval = 100
mm = [[70.7, 633.4], [60.9, 643.4], [60.9, 696.1], [60.9, 696.1]]
flag =int(np.log2(in_dim/64))
transform1 = lambda x: x/255.
transform2 = lambda x: (x - mm[flag][0])/(mm[flag][1] - mm[flag][0])
random_seed = 1
torch.manual_seed(random_seed)
save_path = './checkpoints_imaging{}_{}'.format(in_dim, net)
if not os.path.exists(save_path):
    os.makedirs(save_path)
img_path = './pre_imaging_{}_{}'.format(in_dim, net)
if not os.path.exists(img_path):
    os.makedirs(img_path)
l1_weight = 0.0001
log_path = './log'


m = qqnet(in_dim=in_dim).cuda()
if net == 'uunet':
    m = uunet(in_dim=in_dim).cuda()
if net == 'ggnet':
    m = ggnet(in_dim=in_dim).cuda()
if net == 'qqnet_patch':
    m = qqnet_patch(in_dim=in_dim).cuda()
optim = torch.optim.Adam(m.parameters(), lr=0.001)
loss_f = torch.nn.MSELoss()

train_img_path = r'E:\datasets\plane\plane_crop_train'
train_inten_path = r'E:\projects\SPC_QNN\plane_randomP_intensities\train_dim_{}.npy'.format(in_dim)
test_img_path = r'E:\datasets\plane\plane_crop_test'
test_inten_path = r'E:\projects\SPC_QNN\plane_randomP_intensities\test_dim_{}.npy'.format(in_dim)

test_realimg_path = r'E:\datasets\plane\handmade'
test_realinten_path = r'E:\projects\SPC_QNN\plane_randomP_intensities\handmade_dim_{}.npy'.format(in_dim)


def train(epoch=1000):
    m.train()
    train_dataset = planeDataset(train_img_path, train_inten_path, transform1=transform1, transform2=transform2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    data_num = len(train_dataset)
    for j in range(epoch):
        train_losses = []
        for i, (imgs, buks) in enumerate(train_loader):
            imgs = imgs.to(torch.float32)
            buks = buks.to(torch.float32)
            imgs = imgs.cuda()
            buks = buks.cuda()
            optim.zero_grad()
            y_pre = m(buks)
            loss = loss_f(y_pre, imgs)
            loss.backward()
            optim.step()
            # if i % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         j, i * len(labs), data_num,
            #                100. * i / len(train_loader), loss.item()))
            train_losses.append(loss.item())
        print('----------------Epoch: {} Total train Loss: {:.6f}----------------'.format(
            j, np.mean(train_losses)))
        if j%100 == 0:
            torch.save(m.state_dict(), os.path.join(save_path, str(j)+'_model.pth'))
            torch.save(optim.state_dict(), os.path.join(save_path, str(j)+'_optimizer.pth'))


def test(cpk=50):
    m.load_state_dict(torch.load(os.path.join(save_path, str(cpk)+'_model.pth')))
    m.eval()
    test_dataset = planeDataset(test_img_path, test_inten_path, transform1=transform1, transform2=transform2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)
    test_psnr = 0
    test_ssim = 0
    correct = 0
    with torch.no_grad():
        for i, (imgs, buks) in enumerate(test_loader):
            imgs = imgs.to(torch.float32)
            buks = buks.to(torch.float32)
            buks = buks.cuda()
            pre = m(buks)
            pre1 = pre.cpu().numpy().reshape([64,64])
            cv2.imwrite(img_path+'/epoch{}_{}.jpg'.format(cpk,i), pre1*255)
            imgs = imgs.numpy().reshape([64,64])
            test_psnr += compute_psnr(pre1*255, imgs*255)
            test_ssim += compute_ssim(pre1 * 255, imgs * 255)
    test_psnr /= len(test_dataset)
    test_ssim /= len(test_dataset)
    print('\nTest set: Avg.psnr: {:.4f}  Avg.ssim:{:.4f}\n'.format(test_psnr, test_ssim))

def test_real(cpk=50):
    m.load_state_dict(torch.load(os.path.join(save_path, str(cpk)+'_model.pth')))
    m.eval()
    test_dataset = planeDataset(test_realimg_path, test_realinten_path, transform1=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)
    test_psnr = 0
    test_ssim = 0
    correct = 0
    with torch.no_grad():
        for i, (imgs, buks) in enumerate(test_loader):
            imgs = imgs.to(torch.float32)
            buks = buks.to(torch.float32)
            buks = buks.cuda()
            pre = m(buks)
            pre1 = pre.cpu().numpy().reshape([64,64])
            cv2.imwrite(img_path+'/handmade_epoch{}_{}.jpg'.format(cpk,i), pre1*255)
            imgs = imgs.numpy().reshape([64,64])
            test_psnr += compute_psnr(pre1*255, imgs)
            test_ssim += compute_ssim(pre1 * 255, imgs)
    test_psnr /= len(test_dataset)
    test_ssim /= len(test_dataset)
    print('\nTest set: Avg.psnr: {:.4f}  Avg.ssim:{:.4f}\n'.format(test_psnr, test_ssim))



if __name__ == '__main__':
    log_name = 'plane_{}_indim{}.log'.format(net, in_dim)
    # log_name = 'mnist_10_learnP_{}_in{}_emb{}.log'.format(net, in_dim, embed_dim)
    sys.stdout = Logger(filename=os.path.join(log_path, log_name))
    train(501)
    for i in range(0,1001,100):
       test(cpk=i)