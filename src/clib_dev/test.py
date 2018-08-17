import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.add import MyAddModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

model = MyNetwork()
x = torch.range(1, 25).view(5, 5)
input1, input2 = Variable(x), Variable(x * 4)
print(model(input1, input2))
print(input1 + input2)

if torch.cuda.is_available():
    input1, input2, = input1.cuda(), input2.cuda()
    print(model(input1, input2))
    print(input1 + input2)
