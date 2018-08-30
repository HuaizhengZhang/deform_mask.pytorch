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



def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im