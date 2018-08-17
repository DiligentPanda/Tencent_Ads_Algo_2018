int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
		       THCudaTensor *output);
int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);

int reduce_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output);
int reduce_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input);

int reduce_max_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output, THCudaLongTensor * buffer);
int reduce_max_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input, THCudaLongTensor * buffer);

int reduce_mean_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output);
int reduce_mean_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input);

int replicate_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output);
int replicate_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * grad_input);

int softmax_forward_cuda(THCudaTensor * input, THCudaLongTensor * offsets, THCudaTensor * output);
int softmax_backward_cuda(THCudaTensor * grad_output, THCudaLongTensor * offsets, THCudaTensor * output, THCudaTensor * grad_input);
