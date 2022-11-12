#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> train_clc
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2022/7/18 20:15
@Desc   ：
=================================================='''
import torch
from models import bigNet, QNN_like_2, QNN_like
from data import bucketsDataset
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from torch.optim.lr_scheduler import ExponentialLR
from utils import Logger
import sys

in_dim = 16 #16 32 64 128
net = 'QNN_like'#'QNN_like_2' # 'bigNet' 'QNN_like'
train_flag = 'all' #all 30000 20000 10000 8000 5000 3000 1000 500
batch_size = 256
batch_size_test = 1000
log_interval = 100
random_seed = 1
torch.manual_seed(random_seed)
save_path = './checkpoints_clc_randomP_indim_{}_{}_trainflag_{}'.format(in_dim, net, train_flag) #random P
#save_path = './checkpoints_clc_indim_{}_{}'.format(in_dim, net) #learn P
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = './log'

#mm = [72.6636, 66.8539, 70.2691, 73.5284] #learnP
mm = [18.7773, 20.9703, 20.9703, 23.5209] #randomP
flag =int(np.log2(in_dim/16))
embed_dim = 16
embed_dim_2 = 16
transform = lambda x: x/mm[flag]
m = QNN_like_2(in_dim=in_dim, embed_dim=embed_dim, embed_dim_2=embed_dim_2).cuda()
if net == 'bigNet':
    m = bigNet(in_dim=in_dim).cuda()
if net == 'QNN_like':
    m = QNN_like(in_dim=in_dim).cuda()
optim = torch.optim.Adam(m.parameters(), lr=0.001)
loss_f = torch.nn.NLLLoss()



train_bkt_path = './randomP_intensities/train_dim_{}.npy'.format(in_dim)
#train_bkt_path = './learnP_intensities/train_dim_{}.npy'.format(in_dim)
train_lab_path = r'E:\datasets\mnist\train-labels-idx1-ubyte'
test_bkt_path = './randomP_intensities/test_dim_{}.npy'.format(in_dim)
#test_bkt_path = './learnP_intensities/test_dim_{}.npy'.format(in_dim)
test_lab_path = r'E:\datasets\mnist\test-labels-idx1-ubyte'



def train(epoch=500):
    m.train()
    train_dataset = bucketsDataset(train_bkt_path, train_lab_path, transform=transform, flag=train_flag)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    data_num = len(train_dataset)
    print(data_num)
    for j in range(epoch):
        train_losses = []
        correct = 0
        for i, (bkts, labs) in enumerate(train_loader):
            bkts = bkts.to(torch.float32)
            labs = labs.to(torch.long)
            bkts = bkts.cuda()

            labs = labs.cuda()
            optim.zero_grad()
            _, y_pre = m(bkts)
            loss = loss_f(y_pre, labs)
            loss.backward()
            optim.step()
            pred = y_pre.data.max(1, keepdim=True)[1]
            correct += pred.eq(labs.data.view_as(pred)).sum()
            train_losses.append(loss.item())
            # if i % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         j, i * len(labs), data_num,
            #                100. * i / len(train_loader), loss.item()))

        train_acc = 100. * correct / data_num
        print('Epoch: {} Total train Loss: {:.6f} train acc: {:.6f}'.format(
            j, np.mean(train_losses), train_acc))

        if j%100 == 0:
            torch.save(m.state_dict(), os.path.join(save_path, str(j)+'_model.pth'))
            torch.save(optim.state_dict(), os.path.join(save_path, str(j)+'_optimizer.pth'))


def test(cpk=50):
    m.load_state_dict(torch.load(os.path.join(save_path, str(cpk)+'_model.pth')))
    m.eval()
    test_dataset = bucketsDataset(test_bkt_path, test_lab_path, transform=transform, flag='all')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)
    test_loss = 0
    correct = 0
    features = []
    with torch.no_grad():
        for bkts, labs in test_loader:
            bkts = bkts.to(torch.float32)
            labs = labs.to(torch.long)
            bkts = bkts.cuda()
            labs = labs.cuda()
            fea, pre = m(bkts)
            test_loss += F.nll_loss(pre, labs, reduction='sum').item()
            print(pre.data)
            pred = pre.data.max(1, keepdim=True)[1]
            correct += pred.eq(labs.data.view_as(pred)).sum()
            features.append(fea.cpu().numpy())
    np.save(os.path.join(save_path, str(cpk) + '_test_features.npy'), features)
    test_loss /= len(test_dataset)
    print('[Epoch {}] Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        cpk, test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))

if __name__ == '__main__':
    # log_name = 'mnist_10_randomP_{}_in{}_emb{}_trainflag_{}.log'.format(net, in_dim, embed_dim, train_flag)
    # #log_name = 'mnist_10_learnP_{}_in{}_emb{}.log'.format(net, in_dim, embed_dim)
    # sys.stdout = Logger(filename=os.path.join(log_path, log_name))
    # train(501)
    for i in range(0,501,100):
       test(cpk=i)
