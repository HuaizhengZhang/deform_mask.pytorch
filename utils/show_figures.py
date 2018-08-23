#!/usr/bin/env python
# @Time    : 22/8/18 7:13 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : show_figures.py

import matplotlib.pyplot as plt
import numpy as np

def show_dpsroi_offset(im, boxes, offset, classes, trans_std=0.1):
    plt.cla
    for idx, bbox in enumerate(boxes):
        plt.figure(idx+1)
        plt.axis("off")
        plt.imshow(im)

        offset_w = np.squeeze(offset[idx, classes[idx]*2, :, :]) * trans_std
        offset_h = np.squeeze(offset[idx, classes[idx]*2+1, :, :]) * trans_std
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        roi_width = x2-x1+1
        roi_height = y2-y1+1
        part_size = offset_w.shape[0]
        bin_size_w = roi_width / part_size
        bin_size_h = roi_height / part_size
        show_boxes_simple(bbox, color='b')
        for ih in range(part_size):
            for iw in range(part_size):
                sub_box = np.array([x1+iw*bin_size_w, y1+ih*bin_size_h,
                                    x1+(iw+1)*bin_size_w, y1+(ih+1)*bin_size_h])
                sub_offset = offset_h[ih, iw] * np.array([0, 1, 0, 1]) * roi_height \
                             + offset_w[ih, iw] * np.array([1, 0, 1, 0]) * roi_width
                sub_box = sub_box + sub_offset
                show_boxes_simple(sub_box)
        plt.show()
