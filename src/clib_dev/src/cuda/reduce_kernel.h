#ifndef _REDUCE_KERNEL
#define _REDUCE_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int SoftmaxForwardLauncher(
    const float * input_data,
    const int64_t * offsets_data,
    float * output_data,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int SoftmaxBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets_data,
    const float * output_data,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReduceMaxForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int64_t * buffer,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReduceMaxBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int64_t * buffer,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReduceMeanForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReduceMeanBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);


int ReduceForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReduceBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReplicateForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);

int ReplicateBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
);



#ifdef __cplusplus
}
#endif

#endif