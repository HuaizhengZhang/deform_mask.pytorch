//
// Created by zhz on 8/8/18.
//

#include <THC/THC.h>
#include <math.h>
#include "roi_align_kernel.h"

extern THCState *state;

int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                           THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIAlignForwardLaucher(
            data_flat, spatial_scale, num_rois, data_height,
            data_width, num_channels, aligned_height,
            aligned_width, rois_flat,
            output_flat, stream);

    return 1;
}