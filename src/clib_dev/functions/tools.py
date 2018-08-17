import torch
from torch.autograd import Function,Variable
from clib_dev._ext import my_lib

class MySoftmax(Function):
    @staticmethod
    def forward(ctx, input, offsets):
        '''

        :param ctx:
        :param input:
        :param offsets:
        :return: output: has the same size as input
        '''
        output = torch.zeros_like(input)
        input = input.contiguous()
        offsets = offsets
        if not input.is_cuda:
            my_lib.softmax_forward(input, offsets.contiguous(), output)
        else:
            my_lib.softmax_forward_cuda(input, offsets.contiguous(), output)
        ctx.save_for_backward(offsets,output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_offsets = None
        grad_input = torch.zeros_like(grad_output)
        offsets,output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        offsets = offsets.contiguous()
        output = output.contiguous()
        if not grad_output.is_cuda:
            my_lib.softmax_backward(grad_output.data, offsets, output, grad_input.data)
        else:
            my_lib.softmax_backward_cuda(grad_output.data, offsets, output, grad_input.data)
        return grad_input,grad_offsets

mysoftmax = MySoftmax.apply

class ReduceFunction(Function):
    @staticmethod
    def forward(ctx, input, offsets, mode="sum"):
        ctx.mode = mode
        if mode == "sum":
            MODE = 0
        elif mode == "max":
            MODE = 1
        elif mode == "mean":
            MODE = 2
        else:
            raise NotImplementedError

        # currently we only support sum mode/ only cpu
        input_size = input.size()
        offsets_size = offsets.size()
        n_sample = offsets_size[0]
        n_dim = input_size[1]
        n_feat = input_size[0]
        output = input.new(n_sample,n_dim)
        if MODE==1:
            buffer = input.new(n_sample,n_dim).long()
        # set to 0
        output[:] = 0
        ctx.n_feat = n_feat
        ctx.n_sample = n_sample
        ctx.n_dim = n_dim
        if MODE==0:
            ctx.save_for_backward(offsets)
        elif MODE==1:
            ctx.save_for_backward(offsets)
            ctx.buffer = buffer
        elif MODE==2:
            ctx.save_for_backward(offsets)
        input = input.contiguous()
        offsets = offsets.contiguous()
        if MODE==0:
            if not input.is_cuda:
                my_lib.reduce_forward(input, offsets, output)
            else:
                my_lib.reduce_forward_cuda(input, offsets, output)
        elif MODE==1:
            if not input.is_cuda:
                my_lib.reduce_max_forward(input, offsets, output, buffer)
            else:
                my_lib.reduce_max_forward_cuda(input, offsets, output, buffer)
        elif MODE==2:
            if not input.is_cuda:
                my_lib.reduce_mean_forward(input, offsets, output)
            else:
                my_lib.reduce_mean_forward_cuda(input, offsets, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mode = ctx.mode
        if mode == "sum":
            MODE = 0
        elif mode == "max":
            MODE = 1
        elif mode == "mean":
            MODE = 2
        else:
            raise NotImplementedError
        offsets = ctx.saved_tensors[0]
        if MODE==1:
            buffer = ctx.buffer

        grad_input = grad_offsets = grad_mode = None
        if MODE==0:
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.data.new(ctx.n_feat,ctx.n_dim)
                grad_input[:] = 0
                grad_output = grad_output.contiguous()
                offsets = offsets.contiguous()
                if not grad_output.is_cuda:
                    my_lib.reduce_backward(grad_output.data, offsets, grad_input)
                else:
                    my_lib.reduce_backward_cuda(grad_output.data, offsets, grad_input)
        elif MODE==1:
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.data.new(ctx.n_feat,ctx.n_dim)
                grad_input[:] = 0
                grad_output = grad_output.contiguous()
                offsets = offsets.contiguous()
                buffer = buffer.contiguous()
                if not grad_output.is_cuda:
                    my_lib.reduce_max_backward(grad_output.data, offsets, grad_input,buffer)
                else:
                    my_lib.reduce_max_backward_cuda(grad_output.data, offsets, grad_input,buffer)
        elif MODE==2:
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.data.new(ctx.n_feat,ctx.n_dim)
                grad_input[:] = 0
                grad_output = grad_output.contiguous()
                offsets = offsets.contiguous()
                if not grad_output.is_cuda:
                    my_lib.reduce_mean_backward(grad_output.data, offsets, grad_input)
                else:
                    my_lib.reduce_mean_backward_cuda(grad_output.data, offsets, grad_input)
        #del buffer
        return Variable(grad_input), grad_offsets, grad_mode

reduce = ReduceFunction.apply


class ReplicateFunction(Function):
    @staticmethod
    def forward(ctx, input, offsets, total_len):
        input_size = input.size()
        offsets_size = offsets.size()
        n_sample = offsets_size[0]
        n_dim = input_size[1]
        n_feat = total_len
        output = input.new(n_feat,n_dim)

        # set to 0
        output[:] = 0
        ctx.n_feat = n_feat
        ctx.n_sample = n_sample
        ctx.n_dim = n_dim
        ctx.save_for_backward(offsets)
        input = input.contiguous()
        offsets = offsets.contiguous()
        if not input.is_cuda:
            my_lib.replicate_forward(input, offsets, output)
        else:
            my_lib.replicate_forward_cuda(input, offsets, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        offsets = ctx.saved_tensors[0]
        grad_input = grad_offsets = grad_total_len = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.data.new(ctx.n_sample,ctx.n_dim)
            grad_input[:] = 0
            grad_output = grad_output.contiguous()
            if not grad_output.is_cuda:
                my_lib.replicate_backward(grad_output.data, offsets, grad_input)
            else:
                my_lib.replicate_backward_cuda(grad_output.data, offsets, grad_input)
        return Variable(grad_input), grad_offsets, grad_total_len

replicate = ReplicateFunction.apply
