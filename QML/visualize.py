#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：SPC_QNN -> visualize
@IDE    ：PyCharm
@Author ：ZXL
@Date   ：2022/9/15 14:51
@Desc   ：tsne
=================================================='''
from utils import tsne
import numpy as np
import struct
import matplotlib.pyplot as plt

dim = 32
net = 'QNN_like_2'

####################################################
path = './learnP_intensities/test_dim_{}.npy'.format(dim)
#path = './checkpoints_clc_randomP_indim_{}_{}/500_test_features.npy'.format(dim, net)
#####################################################

label_path = r'E:\datasets\mnist\test-labels-idx1-ubyte'
with open(label_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)
feas = np.load(path)

###################################################
#feas = np.reshape(feas, (feas.shape[0]*feas.shape[1], feas.shape[2]))
###################################################

print(feas.shape)

########################################################
save_name = './visualization/learnP_origin_indim_{}.npy'.format(dim)
# save_name = './visualization/randomP_{}_indim_{}.npy'.format(net, dim)
#######################################################

a = tsne(feas)
np.save(save_name, a)
#a = np.array(np.load(save_name))
a = (a - np.min(a))/(np.max(a)-np.min(a))
colors = ['red','orange','yellow','green','cyan','blue','purple','pink','magenta','brown']
b = [[i for i, x in enumerate(labels) if int(x) == j] for j in range(10)]
for i in range(10):
    plt.scatter(a[b[i], 0], a[b[i], 1], c=colors[i], s=5, marker='o', alpha=0.5,edgecolors='none', label=i)
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Main component 1')
plt.ylabel('Main component 2')

#################################################################
plt.savefig('./visualization/learnP_origin_indim_{}.svg'.format(dim))
#plt.savefig('./visualization/randomP_{}_indim_{}.svg'.format(net, dim))
#################################################################

plt.show()

