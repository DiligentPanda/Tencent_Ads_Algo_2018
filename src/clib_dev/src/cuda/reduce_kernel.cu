#ifdef __cpluscplus
extern "C"{
#endif

#include "reduce_kernel.h"
#include <stdio.h>

#define CUDA_1D_KERNEL_LOOP(i,n)                        \
    for (int i=blockIdx.x*blockDim.x + threadIdx.x;i<n; \
        i += blockDim.x * gridDim.x)


__global__ void SoftmaxForward(
    const int nthreads,
    const float * input_data,
    const int64_t * offsets_data,
    float * output_data,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int k = index%n_dim; // dim_idx
        int s = offsets_data[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets_data[sample_idx+1];

         // get max
        float v_max = input_data[s*n_dim+k];
        for (int j=s+1;j<e;++j)
        {
            if (input_data[j*n_dim+k]>v_max)
            {
                v_max = input_data[j*n_dim+k];
            }
        }

        // subtract max and exp, accumulate sum
        float sum = 0;
        for (int j=s;j<e;++j)
        {
            output_data[j*n_dim+k] = exp(input_data[j*n_dim+k]-v_max);
            sum += output_data[j*n_dim+k];
        }

        // divide sum
        for (int j=s;j<e;++j)
        {
            output_data[j*n_dim+k] /= sum;
        }

    }

}


__global__ void ReduceMaxForward(
    const int nthreads,
    const float * input,
    const int64_t * offsets,
    float * output,
    int64_t * buffer,
    const int n_feat,
    const int n_dim,
    const int n_sample)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        output[index] = input[s*n_dim+dim_idx];
        buffer[index] = s*n_dim+dim_idx;
        for (int i=s;i<e;++i)
        {
               int input_idx = i*n_dim+dim_idx;
               if (input[input_idx]>output[index])
               {
                    output[index] = input[input_idx];
                    buffer[index] = input_idx;
               }
        }
    }

}

__global__ void ReduceMeanForward(
    const int nthreads,
    const float * input,
    const int64_t * offsets,
    float * output,
    const int n_feat,
    const int n_dim,
    const int n_sample)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int input_idx = i*n_dim+dim_idx;
               output[index] += input[input_idx];
        }
        output[index] = output[index]/(e-s);
    }

}


__global__ void ReduceForward(
    const int nthreads,
    const float * input,
    const int64_t * offsets,
    float * output,
    const int n_feat,
    const int n_dim,
    const int n_sample)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int input_idx = i*n_dim+dim_idx;
               output[index] += input[input_idx];
        }
    }

}


int SoftmaxForwardLauncher(
    const float * input_data,
    const int64_t * offsets_data,
    float * output_data,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    SoftmaxForward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, input_data, offsets_data, output_data, n_feat, n_dim, n_sample);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


int ReduceMaxForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int64_t * buffer,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceMaxForward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, input, offsets, output, buffer, n_feat, n_dim, n_sample);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


int ReduceMeanForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceMeanForward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, input, offsets, output, n_feat, n_dim, n_sample);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


int ReduceForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceForward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, input, offsets, output, n_feat, n_dim, n_sample);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


__global__ void SoftmaxBackward(
    const int nthreads,
    const float * output_grad,
    const int64_t * offsets_data,
    const float * output_data,
    float * input_grad,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int k = index%n_dim; // dim_idx
        int s = offsets_data[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets_data[sample_idx+1];

        for (int j=s;j<e;++j)
        {
            input_grad[j*n_dim+k] = output_grad[j*n_dim+k];
            for (int l=s;l<e;++l)
            {
                input_grad[j*n_dim+k] -= output_grad[l*n_dim+k]*output_data[l*n_dim+k];
            }
            input_grad[j*n_dim+k] *= output_data[j*n_dim+k];
        }
    }

}


__global__ void ReduceMaxBackward(
    const int nthreads,
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int64_t * buffer,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        //int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int input_idx = buffer[index];
               input_grad[input_idx] = output_grad[index];
        }
    }

}

__global__ void ReduceMeanBackward(
    const int nthreads,
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int input_idx = i*n_dim+dim_idx;
               input_grad[input_idx] = output_grad[index]/(e-s);
        }
    }

}

__global__ void ReduceBackward(
    const int nthreads,
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int input_idx = i*n_dim+dim_idx;
               input_grad[input_idx] = output_grad[index];
        }
    }

}


int SoftmaxBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets_data,
    const float * output_data,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    SoftmaxBackward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, output_grad, offsets_data, output_data, input_grad, n_feat, n_dim, n_sample
    );

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


int ReduceMaxBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int64_t * buffer,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceMaxBackward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, output_grad, offsets, input_grad, buffer, n_feat, n_dim, n_sample
    );

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}

int ReduceMeanBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceMeanBackward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, output_grad, offsets, input_grad, n_feat, n_dim, n_sample
    );

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


int ReduceBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReduceBackward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, output_grad, offsets, input_grad, n_feat, n_dim, n_sample
    );

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


__global__ void ReplicateForward(
    const int nthreads,
    const float * input,
    const int64_t * offsets,
    float * output,
    const int n_feat,
    const int n_dim,
    const int n_sample)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int o_idx = i*n_dim+dim_idx;
               output[o_idx] = input[index];
        }
    }

}

int ReplicateForwardLauncher(
    const float * input,
    const int64_t * offsets,
    float * output,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream
)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReplicateForward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, input, offsets, output, n_feat, n_dim, n_sample);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}

__global__ void ReplicateBackward(
    const int nthreads,
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    const int n_feat,
    const int n_dim,
    const int n_sample
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        int sample_idx = index/n_dim;
        int dim_idx = index%n_dim;
        int s = offsets[sample_idx];
        int e = -1;
        if (sample_idx==n_sample-1) e = n_feat;
        else e = offsets[sample_idx+1];
        for (int i=s;i<e;++i)
        {
               int o_idx = i*n_dim+dim_idx;
               input_grad[index] += output_grad[o_idx];
        }
    }
}


int ReplicateBackwardLauncher(
    const float * output_grad,
    const int64_t * offsets,
    float * input_grad,
    int n_feat,
    int n_dim,
    int n_sample,
    cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int nthreads = n_sample * n_dim;

    cudaError_t err;

    // call cuda func
    ReplicateBackward<<<(nthreads+kThreadsPerBlock-1)/kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
        nthreads, output_grad, offsets, input_grad, n_feat, n_dim, n_sample
    );

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;

}


#ifdef __cpluscplus
extern }
#endif