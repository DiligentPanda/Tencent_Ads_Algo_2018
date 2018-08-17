from torch.autograd import Variable
from functions.tools import reduce,replicate
import torch

from torch.autograd import gradcheck

input = Variable(torch.Tensor([[7, 2], [4, 3], [5, 6]]).cuda())
offsets = Variable(torch.LongTensor([0, 2]).cuda())
results = reduce(input, offsets, "mean")
print(results)

# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

input = (
    Variable(torch.Tensor([[7, 2], [4, 3], [5, 6]]).cuda(), requires_grad=True),
    Variable(torch.LongTensor([0,2]).cuda()),
)
print(input)
test = gradcheck(reduce, input, eps=1e-3, atol=1e-3,rtol=1e-3)
print(test)
# there is a strange bug if call cuda() after Variable, then variable would be marked as not leaf
# and no grad is retained.
output = reduce(input[0],input[1],"mean")
output.backward(torch.Tensor([[1,4],[1,2]]).cuda())
print(input[0].grad)


# input = Variable(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
# offsets = Variable(torch.LongTensor([0, 2]))
# total_len = 5
# results = replicate(input, offsets, total_len)
# print(results)
#
# input = (
#     Variable(torch.Tensor([[1, 2, 3], [4, 5, 6]]).cuda(), requires_grad=True),
#     Variable(torch.LongTensor([0,2]).cuda()),
# )
# print(input)
# test = gradcheck(reduce, input, eps=1e-4, atol=1e-3,rtol=1e-3)
# print(test)