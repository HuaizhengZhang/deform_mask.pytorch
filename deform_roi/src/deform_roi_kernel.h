#ifndef _DEFORM_ROI_KERNEL
#define _DEFORM_ROI_KERNEL

#ifdef __cplusplus
extern "C" {
#endif



int DeformPSROIPoolForwardLauncher (
            const float* bottom_data, const float spatial_scale, const int num_rois,
            const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
            const float* bottom_rois, const float* bottom_trans, const bool no_trans, const float trans_std,
            const int sample_per_part, const int output_dim, const int num_classes, const int group_size,
            const int part_size, float* top_data, float* mapping_channel, cudaStream_t stream);



int DeformPSROIPoolBackwardLauncher(
        const float* top_diff, const float* mapping_channel, const int num_rois, const float spatial_scale,
        const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
        const int output_dim, float* bottom_data_diff, float* bottom_trans_diff, const float* bottom_data,
        const float* bottom_rois, const float* bottom_trans, const bool no_trans, const float trans_std,
        const int sample_per_part, const int group_size, const int part_size,
        const int num_classes, const int channels_each_class, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

