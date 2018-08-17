import torch
from clib_dev.functions.tools import reduce,replicate,mysoftmax

# #@timethis
# def reduce(input, offsets,mode="sum"):
#     '''
#     :param input: [ n_feat, n_dim ]
#     :param offsets: [ n_sample ]
#     :return: [n_sample, n_dim]
#     '''
#
#     input_size = input.size()
#     offsets_data = offsets.data
#     n_sample = len(offsets_data)
#     results = []
#     for i in range(n_sample):
#         s = offsets_data[i]
#         e = offsets_data[i + 1] if i < n_sample - 1 else input_size[0]
#         if mode=="sum":
#             result = torch.sum(input[s:e,:],dim=0)
#         elif mode=="mean":
#             result = torch.mean(input[s:e,:],dim=0)
#         elif mode=="max":
#             result,_ = torch.max(input[s:e,:],dim=0)
#         else:
#             raise Exception("Unknown mode")
#         results.append(result)
#     results = torch.stack(results,dim=0)
#     return results
#
# #@timethis
# def replicate(input, offsets, total_len):
#     '''
#     :param input: [ n_sample, n_dim ]
#     :param offsets: [ n_sample ]
#     :param total_len: the len of results
#     :return: [total_len, n_dim]
#     '''
#     #input_size = input.size()
#     offsets_data = offsets.data
#     n_sample = len(offsets_data)
#     results = []
#     for i in range(n_sample):
#         s = offsets_data[i]
#         e = offsets_data[i + 1] if i < n_sample - 1 else total_len
#         result = input[i,:].view(1,-1).repeat(e-s,1)
#         #print(result)
#         results.append(result)
#     #print(results)
#     results = torch.cat(results,dim=0)
#     return results


if __name__=="__main__":
    from torch.autograd import Variable
    input = Variable(torch.Tensor([[1,2],[4,3],[5,6]]))
    offsets = Variable(torch.LongTensor([0,2]))
    results = reduce(input,offsets)
    print(results)

    input = Variable(torch.Tensor([[1,2,3],[4,5,6]]))
    offsets = Variable(torch.LongTensor([0,2]))
    total_len = 5
    results = replicate(input,offsets,total_len)
    print(results)