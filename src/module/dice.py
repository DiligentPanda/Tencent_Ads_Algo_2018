import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self,dim,init=0):
        super(Dice,self).__init__()
        self.dim = dim
        # todo the momentum in paper is weird, we use the default choice here
        # todo maybe affine?
        self.bn = nn.BatchNorm1d(dim,momentum=0.01,affine=False)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(self.dim)*init)

    def forward(self, x):
        y = self.bn(x)
        p = self.sigmoid(y)
        a = self.alpha*(1-p)*x + p*x
        return a