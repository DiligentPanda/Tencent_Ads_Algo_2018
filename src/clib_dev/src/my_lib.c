#include <TH/TH.h>
#include <math.h>

int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output)
{
  if (!THFloatTensor_isSameSizeAs(input1, input2))
    return 0;
  THFloatTensor_resizeAs(output, input1);
  THFloatTensor_cadd(output, input1, 1.0, input2);
  return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
  THFloatTensor_resizeAs(grad_input, grad_output);
  THFloatTensor_fill(grad_input, 1);
  return 1;
}

int softmax_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output)
{
    int64_t n_feat = THFloatTensor_size(input, 0);
    int64_t n_dim = THFloatTensor_size(input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);

    const int64_t * offsets_data = THLongTensor_data(offsets);
    const float * input_data = THFloatTensor_data(input);
    float * output_data = THFloatTensor_data(output);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int k=0;k<n_dim;++k)
        {
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

    return 1;
}


int softmax_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * output, THFloatTensor * grad_input)
{
    int64_t n_feat = THFloatTensor_size(grad_input, 0);
    int64_t n_dim = THFloatTensor_size(grad_input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);

    const int64_t * offsets_data = THLongTensor_data(offsets);
    const float * output_data = THFloatTensor_data(output);
    const float * grad_output_data = THFloatTensor_data(grad_output);
    float * grad_input_data = THFloatTensor_data(grad_input);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int k=0;k<n_dim;++k)
        {
             for (int j=s;j<e;++j)
             {
                grad_input_data[j*n_dim+k] = grad_output_data[j*n_dim+k];
                for (int l=s;l<e;++l)
                {
                    grad_input_data[j*n_dim+k] -= grad_output_data[l*n_dim+k]*output_data[l*n_dim+k];
                }
                grad_input_data[j*n_dim+k] *= output_data[j*n_dim+k];
             }
        }
    }
    return 1;
}



int reduce_max_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output, THLongTensor * buffer)
{
    int64_t n_feat = THFloatTensor_size(input, 0);
    int64_t n_dim = THFloatTensor_size(input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);
    const int64_t * offsets_data = THLongTensor_data(offsets);
    const float * input_data = THFloatTensor_data(input);
    float * output_data = THFloatTensor_data(output);
    int64_t * buffer_data = THLongTensor_data(buffer);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            output_data[o_idx] = input_data[s*n_dim+j];
            buffer_data[o_idx] = s*n_dim+j;
            for (int k=s+1;k<e;++k)
            {
                int i_idx = k*n_dim+j;
                if (input_data[i_idx]>output_data[o_idx])
                {
                    output_data[o_idx] = input_data[i_idx];
                    buffer_data[o_idx] = i_idx;
                }
            }
        }
    }
    return 1;
}

int reduce_max_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input, THLongTensor * buffer)
{
    int64_t n_feat = THFloatTensor_size(grad_input, 0);
    int64_t n_dim = THFloatTensor_size(grad_input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);

    const int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_grad = THFloatTensor_data(grad_input);
    const float * output_grad = THFloatTensor_data(grad_output);
    const int64_t * buffer_data = THLongTensor_data(buffer);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int i_idx = buffer_data[o_idx];
                input_grad[i_idx] = output_grad[o_idx];
            }
        }
    }
    return 1;
}

// we follow the codes above ret 0 means sth wrong
int reduce_mean_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output)
{
    int64_t n_feat = THFloatTensor_size(input, 0);
    int64_t n_dim = THFloatTensor_size(input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);
    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_data = THFloatTensor_data(input);
    float * output_data = THFloatTensor_data(output);
    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int i_idx = k*n_dim+j;
                output_data[o_idx] += input_data[i_idx];
            }
            output_data[o_idx] = output_data[o_idx]/(e-s);
        }
    }
    return 1;
}

int reduce_mean_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input)
{
    int64_t n_feat = THFloatTensor_size(grad_input, 0);
    int64_t n_dim = THFloatTensor_size(grad_input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);

    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_grad = THFloatTensor_data(grad_input);
    float * output_grad = THFloatTensor_data(grad_output);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int i_idx = k*n_dim+j;
                input_grad[i_idx] = output_grad[o_idx]/(e-s);
            }
        }
    }
    return 1;
}



// we follow the codes above ret 0 means sth wrong
int reduce_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output)
{
    int64_t n_feat = THFloatTensor_size(input, 0);
    int64_t n_dim = THFloatTensor_size(input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);
    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_data = THFloatTensor_data(input);
    float * output_data = THFloatTensor_data(output);
    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int i_idx = k*n_dim+j;
                output_data[o_idx] += input_data[i_idx];
            }
        }
    }
    return 1;
}

int reduce_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input)
{
    int64_t n_feat = THFloatTensor_size(grad_input, 0);
    int64_t n_dim = THFloatTensor_size(grad_input, 1);
    int64_t n_sample = THLongTensor_size(offsets, 0);

    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_grad = THFloatTensor_data(grad_input);
    float * output_grad = THFloatTensor_data(grad_output);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int o_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int i_idx = k*n_dim+j;
                input_grad[i_idx] = output_grad[o_idx];
            }
        }
    }
    return 1;
}

// we follow the codes above ret 0 means sth wrong
int replicate_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output)
{
    int64_t n_sample = THFloatTensor_size(input, 0);
    int64_t n_dim = THFloatTensor_size(input, 1);
    int64_t n_feat = THFloatTensor_size(output, 0);
    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_data = THFloatTensor_data(input);
    float * output_data = THFloatTensor_data(output);
    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int i_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int o_idx = k*n_dim+j;
                output_data[o_idx] = input_data[i_idx];
            }
        }
    }
    return 1;
}

int replicate_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input)
{
    int64_t n_sample = THFloatTensor_size(grad_input, 0);
    int64_t n_dim = THFloatTensor_size(grad_input, 1);
    int64_t n_feat = THFloatTensor_size(grad_output, 0);

    int64_t * offsets_data = THLongTensor_data(offsets);
    float * input_grad = THFloatTensor_data(grad_input);
    float * output_grad = THFloatTensor_data(grad_output);

    for (int i=0;i<n_sample;++i)
    {
        int64_t s = offsets_data[i];
        int64_t e = i<n_sample-1?offsets_data[i+1]:n_feat;
        for (int j=0;j<n_dim;++j)
        {
            int i_idx = i*n_dim+j;
            for (int k=s;k<e;++k)
            {
                int o_idx = k*n_dim+j;
                input_grad[i_idx] += output_grad[o_idx];
            }
        }
    }
    return 1;
}