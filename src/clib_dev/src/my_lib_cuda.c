#include <THC/THC.h>
#include <stdio.h>
#include "cuda/reduce_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
		       THCudaTensor *output)
{
  if (!THCudaTensor_isSameSizeAs(state, input1, input2))
    return 0;
  THCudaTensor_resizeAs(state, output, input1);
  THCudaTensor_cadd(state, output, input1, 1.0, input2);
  return 1;
}

int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{
  THCudaTensor_resizeAs(state, grad_input, grad_output);
  THCudaTensor_fill(state, grad_input, 1);
  return 1;
}

int softmax_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output)
{
    int64_t n_feat = THCudaTensor_size(state,input, 0);
    int64_t n_dim = THCudaTensor_size(state,input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);

    const int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    const float * input_data = THCudaTensor_data(state,input);
    float * output_data = THCudaTensor_data(state,output);

    cudaStream_t stream = THCState_getCurrentStream(state);

    // call forward kernel
    SoftmaxForwardLauncher(
    input_data,
    offsets_data,
    output_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}


int softmax_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * output, THCudaTensor * grad_input)
{
    int64_t n_feat = THCudaTensor_size(state,grad_input, 0);
    int64_t n_dim = THCudaTensor_size(state,grad_input, 1);
    int64_t n_sample = THCudaLongTensor_size(state, offsets, 0);

    const float * output_data = THCudaTensor_data(state,output);
    const int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    const float * output_grad = THCudaTensor_data(state,grad_output);
    float * input_grad = THCudaTensor_data(state,grad_input);

    cudaStream_t stream = THCState_getCurrentStream(state);

    // call backward kernel
    SoftmaxBackwardLauncher(
    output_grad,
    offsets_data,
    output_data,
    input_grad,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}


int reduce_max_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output, THCudaLongTensor *buffer)
{
    int64_t n_feat = THCudaTensor_size(state,input, 0);
    int64_t n_dim = THCudaTensor_size(state,input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);
    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_data = THCudaTensor_data(state,input);
    float * output_data = THCudaTensor_data(state,output);
    int64_t * buffer_data = THCudaLongTensor_data(state,buffer);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call forward kernel
    ReduceMaxForwardLauncher(
    input_data,
    offsets_data,
    output_data,
    buffer_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}

int reduce_max_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input, THCudaLongTensor * buffer)
{
    int64_t n_feat = THCudaTensor_size(state,grad_input, 0);
    int64_t n_dim = THCudaTensor_size(state,grad_input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);

    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_grad = THCudaTensor_data(state,grad_input);
    float * output_grad = THCudaTensor_data(state,grad_output);
    int64_t * buffer_data = THCudaLongTensor_data(state,buffer);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call backward kernel
    ReduceMaxBackwardLauncher(
    output_grad,
    offsets_data,
    input_grad,
    buffer_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}


int reduce_mean_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output)
{
    int64_t n_feat = THCudaTensor_size(state,input, 0);
    int64_t n_dim = THCudaTensor_size(state,input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);
    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_data = THCudaTensor_data(state,input);
    float * output_data = THCudaTensor_data(state,output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call forward kernel
    ReduceMeanForwardLauncher(
    input_data,
    offsets_data,
    output_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}

int reduce_mean_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input)
{
    int64_t n_feat = THCudaTensor_size(state,grad_input, 0);
    int64_t n_dim = THCudaTensor_size(state,grad_input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);

    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_grad = THCudaTensor_data(state,grad_input);
    float * output_grad = THCudaTensor_data(state,grad_output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call backward kernel
    ReduceMeanBackwardLauncher(
    output_grad,
    offsets_data,
    input_grad,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}




int reduce_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output)
{
    int64_t n_feat = THCudaTensor_size(state,input, 0);
    int64_t n_dim = THCudaTensor_size(state,input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);
    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_data = THCudaTensor_data(state,input);
    float * output_data = THCudaTensor_data(state,output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call forward kernel
    ReduceForwardLauncher(
    input_data,
    offsets_data,
    output_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}

int reduce_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input)
{
    int64_t n_feat = THCudaTensor_size(state,grad_input, 0);
    int64_t n_dim = THCudaTensor_size(state,grad_input, 1);
    int64_t n_sample = THCudaLongTensor_size(state,offsets, 0);

    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_grad = THCudaTensor_data(state,grad_input);
    float * output_grad = THCudaTensor_data(state,grad_output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call backward kernel
    ReduceBackwardLauncher(
    output_grad,
    offsets_data,
    input_grad,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}

int replicate_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output)
{
    int64_t n_sample = THCudaTensor_size(state,input, 0);
    int64_t n_dim = THCudaTensor_size(state,input, 1);
    int64_t n_feat = THCudaTensor_size(state,output, 0);

    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_data = THCudaTensor_data(state,input);
    float * output_data = THCudaTensor_data(state,output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call forward kernel
    ReplicateForwardLauncher(
    input_data,
    offsets_data,
    output_data,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}

int replicate_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input)
{
    int64_t n_sample = THCudaTensor_size(state,grad_input, 0);
    int64_t n_dim = THCudaTensor_size(state,grad_input, 1);
    int64_t n_feat = THCudaTensor_size(state,grad_output, 0);

    int64_t * offsets_data = THCudaLongTensor_data(state,offsets);
    float * input_grad = THCudaTensor_data(state,grad_input);
    float * output_grad = THCudaTensor_data(state,grad_output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    // call backward kernel
    ReplicateBackwardLauncher(
    output_grad,
    offsets_data,
    input_grad,
    n_feat,
    n_dim,
    n_sample,
    stream);

    return 1;
}