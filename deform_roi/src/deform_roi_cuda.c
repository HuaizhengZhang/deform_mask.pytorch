//
// Created by zhz on 8/8/18.
//
#include <stdbool.h>
#include <THC/THC.h>
#include <math.h>
#include "deform_roi_kernel.h"

extern THCState *state;

int deform_roi_forward_cuda(int pool_height, int pool_width, float spatial_scale,
                            bool no_trans, float trans_std,
                            int sample_per_part, int output_dim, int group_size,
                            int part_size,
                            THCudaTensor * bottom_trans,
                            THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output,
                            THCudaTensor * mapping_channel
                            )
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



    // Add deform part
    float * bottom_trans_flat = THCudaTensor_data(state, bottom_trans);
    int num_bottom_trans = THCudaTensor_size(state, bottom_trans, 1) / 2;
    int num_class =  no_trans ? 1 : num_bottom_trans;

    float * mapping_channel_flat = THCudaTensor_data(state, mapping_channel);

    cudaStream_t stream = THCState_getCurrentStream(state);

    DeformPSROIPoolForwardLauncher (
            data_flat, spatial_scale, num_rois,
            num_channels, data_height, data_width, pool_height,
            pool_width, rois_flat, bottom_trans_flat, no_trans, trans_std,
            sample_per_part, output_dim, num_class, group_size,
            part_size, output_flat, mapping_channel_flat, stream);

    return 1;
}


int deform_roi_backward_cuda(int pool_height, int pool_width, float spatial_scale,
                             int output_dim, bool no_trans, float trans_std,
                             int sample_per_part, int group_size, int part_size,
                             THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad,
                             THCudaTensor * mapping_channel, THCudaTensor * bottom_trans_grad,
                             THCudaTensor * features, THCudaTensor * bottom_trans)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
//    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    //add deform
    float * mapping_channel_flat = THCudaTensor_data(state, mapping_channel);
    float * bottom_trans_grad_flat =  THCudaTensor_data(state, bottom_trans_grad);
    // TO-DO: features? output?
    float * data_flat = THCudaTensor_data(state, features);

    float * bottom_trans_flat = THCudaTensor_data(state, bottom_trans);
    int num_bottom_trans = THCudaTensor_size(state, bottom_trans, 1) / 2;
    int num_class =  no_trans ? 1 : num_bottom_trans;

    int channels_each_class = output_dim / num_class;

    cudaStream_t stream = THCState_getCurrentStream(state);

    DeformPSROIPoolBackwardLauncher(
            top_grad_flat, mapping_channel_flat, num_rois, spatial_scale,
            num_channels, data_height, data_width, pool_height, pool_width,
            output_dim, bottom_grad_flat, bottom_trans_grad_flat, data_flat,
            rois_flat, bottom_trans_flat, no_trans, trans_std,
            sample_per_part, group_size, part_size,
            num_class, channels_each_class, stream);

    return 1;
}
