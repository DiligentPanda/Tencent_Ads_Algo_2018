int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output);
int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

int reduce_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output);
int reduce_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input);

int reduce_mean_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output);
int reduce_mean_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input);

int reduce_max_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output, THLongTensor * buffer);
int reduce_max_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input, THLongTensor * buffer);

int replicate_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output);
int replicate_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * grad_input);

int softmax_forward(THFloatTensor * input, THLongTensor * offsets, THFloatTensor * output);
int softmax_backward(THFloatTensor * grad_output, THLongTensor * offsets, THFloatTensor * output, THFloatTensor * grad_input);