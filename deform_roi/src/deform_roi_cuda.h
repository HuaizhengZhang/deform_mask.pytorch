//
// Created by zhz on 8/8/18.
//

int deform_roi_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                            int no_trans, float trans_std,
                            int sample_per_part, int output_dim, int group_size,
                            int part_size,
                            THCudaTensor * bottom_trans,
                            THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output,
                            THCudaTensor * mapping_channel
);

int deform_roi_backward_cuda(int pool_height, int pool_width, float spatial_scale,
                             int output_dim, int no_trans, float trans_std,
                             int sample_per_part, int group_size, int part_size,
                             THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad,
                             THCudaTensor * mapping_channel, THCudaTensor * bottom_trans_grad,
                             THCudaTensor * features, THCudaTensor * bottom_trans);
