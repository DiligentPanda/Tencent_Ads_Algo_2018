import torch
import torch.nn as nn

def bce_focal_loss(input, target, gamma=2, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    prob = torch.sigmoid(input)
    prob_for_gt = target*prob+(1-target)*(1-prob)

    if weight is not None:
        loss = loss * weight

    loss = loss * torch.pow((1-prob_for_gt),gamma)
    #print(torch.pow((1-prob_for_gt),gamma))

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

if __name__=="__main__":
    from torch.autograd import Variable
    input = Variable(torch.Tensor([-2.197,0.0,2.197]))
    target = Variable(torch.Tensor([0,0,1]))
    loss = bce_focal_loss(input,target)
    print(loss)