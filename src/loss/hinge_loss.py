import torch

def hinge_loss(input,target,margin=1,weight=None,size_average=True):
    '''

    :param input:
    :param target: assume to be {0,1}
    :param margin:
    :param weight:
    :param size_average:
    :return:
    '''
    target = 2*target-1
    l = torch.max(margin-input*target,torch.zeros_like(target))
    if weight is not None:
        l = l * weight
    if size_average:
        l = torch.mean(l)
    else:
        l = torch.sum(l)
    return l

if __name__=="__main__":
    from torch.autograd import Variable
    x = Variable(torch.FloatTensor([0.5,1.7,1.0]))
    y = Variable(torch.FloatTensor([1,1,0]))
    l = hinge_loss(x,y,size_average=False)
    print(l)
