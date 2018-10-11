#!/usr/bin/env python
# @Time    : 10/10/18 6:12 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : vis.py

import yaml
import torch
from dataloader.relationship import *
import matplotlib.pyplot as plt
from scipy.misc import imread
import cv2
import numpy as np


with open('cfgs/vrd.yml', 'r') as f:
    data_opts = yaml.load(f)

train_test = 'test'
number = 5
train_set = pre_VRD(data_opts, train_test)

imdb = train_set.imdb('data/cache')

# test_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
#                                                 shuffle=False, collate_fn=VRD.collate)

fig = plt.figure()

real = train_set.annotations[number]
rescale = train_set[number]

print(real['relationships'])
print(rescale['relations'])

sample = 'data/VRD/sg_dataset/sg_{}_images/{}'.format(train_test, real['path'])
im = imread(sample)


for i in real['objects']:
    bbox = tuple(int(np.round(x)) for x in i['bbox'])
    cv2.rectangle(im, bbox[0:2], bbox[2:4], (200, 0, 0), 2)


plt.imshow(im)
plt.show()