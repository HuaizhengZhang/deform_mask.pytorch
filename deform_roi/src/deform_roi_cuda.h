//
// Created by zhz on 8/8/18.
//

#ifndef SRC_DEFORM_ROI_CUDA_H
#define SRC_DEFORM_ROI_CUDA_H

#endif //SRC_DEFORM_ROI_CUDA_H



int roi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                             THCudaTensor * features, THCudaTensor * rois,
                             THCudaTensor * output, THCudaIntTensor * argmax);

int roi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                              THCudaTensor * top_grad, THCudaTensor * rois,
                              THCudaTensor * bottom_grad, THCudaIntTensor * argmax);
