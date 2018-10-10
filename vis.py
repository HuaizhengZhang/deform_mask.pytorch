#!/usr/bin/env python
# @Time    : 10/10/18 6:12 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : vis.py

import yaml
from dataloader.relationship import VRD
import matplotlib.pyplot as plt
from scipy.misc import imread
import cv2
import numpy as np


with open('cfgs/vrd.yml', 'r') as f:
    data_opts = yaml.load(f)

train_set = VRD(data_opts, 'test', batch_size=1)


fig = plt.figure()

real = train_set.annotations[0]
rescale = train_set[0]

print(real['relationships'])
print(rescale['relations'])

sample = 'data/VRD/sg_dataset/sg_test_images/' + real['path']
im = imread(sample)


for i in real['objects']:
    bbox = tuple(int(np.round(x)) for x in i['bbox'])
    cv2.rectangle(im, bbox[0:2], bbox[2:4], (200, 0, 0), 2)


plt.imshow(im)
plt.show()