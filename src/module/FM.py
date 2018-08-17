import torch
from torch.nn import Module,Parameter,Tanh
import torch.nn as nn

def cross_mul(x, y=None, idices=None):
        '''

        :param x: (B,F,D)
        :param y: (B,D,F)
        :return:
        '''
        B,F,D = x.size()
        if y is None:
            y = x.transpose(1,2)
        z = torch.matmul(x,y)
        z = z.view(B,F*F)
        if idices is None:
            # generate mask to select features
            j = torch.arange(F).view(1,-1).repeat(F,1)
            i = torch.arange(F).view(-1,1).repeat(1,F)
            mask = (i<=j).view(-1)
            idices = torch.nonzero(mask)[:,0]
        r = z[:,idices]
        return r

def out_product(x, y):
    '''

    :param x: (B,F)
    :param y: (B,F)
    :return: z: (B,F*F)
    '''
    B,F = x.size()
    x = x.view(B,-1,F)
    y = y.view(B,F,-1)
    z = x*y
    return z

class OutProductFeature(Module):
    def __init__(self):
        super(OutProductFeature,self).__init__()

    def forward(self, x, y):
        B,F = x.size()
        z = out_product(x,y)
        z = z.view(B,F*F)
        return z

class FM(Module):
    def __init__(self, n_feat):
        super(FM,self).__init__()
        F = n_feat
        self.n_feat = n_feat
        j = torch.arange(F).view(1, -1).repeat(F, 1)
        i = torch.arange(F).view(-1, 1).repeat(1, F)
        mask = (i <= j).view(-1)
        idices = torch.nonzero(mask)[:, 0]
        self.register_buffer('idices',idices)

    def forward(self, x):
        '''

        :param x: [B,F,D]
        :return: r [B, F*(F-1)/2]
        '''
        r = cross_mul(x,idices=self.idices)
        return r

    def reset_parameters(self):
        pass

class FFMSimple(Module):
    '''
    This is a simplified version of FFM.
    '''
    def __init__(self,n_feat,embed_size):
        super(FFMSimple,self).__init__()
        self.n_feat = n_feat
        # generate idices
        self.x_idices = []
        for i in range(self.n_feat):
            for j in range(self.n_feat-i):
                self.x_idices.append(i)
        self.y_idices = []
        for i in range(self.n_feat):
            for j in range(i,self.n_feat):
                self.y_idices.append(j)
        self.weight = Parameter(torch.Tensor(n_feat*(n_feat+1)//2,embed_size))
        self.activation = Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight,1)

    def forward(self, input):
        '''

        :param input: B,F,
        :return:
        '''

        x = input[:,self.x_idices,:]
        y = input[:,self.y_idices,:]

        r = torch.sum(x*self.activation(self.weight)*y,dim=2)

        return r

class FFM(Module):
    def __init__(self,n_feat):
        super(FFM,self).__init__()
        self.n_feat = n_feat
        # generate idices
        x_idices = []
        y_idices = []
        for i in range(0,self.n_feat):
            for j in range(i,self.n_feat):
                x_idices.append(i*self.n_feat+j)
        for j in range(0,self.n_feat):
            for i in range(j,self.n_feat):
                y_idices.append(i*self.n_feat+j)
        x_idices = torch.LongTensor(x_idices)
        y_idices = torch.LongTensor(y_idices)
        self.register_buffer("x_idices",x_idices)
        self.register_buffer("y_idices", y_idices)
        print(self.x_idices)
        print(len(self.y_idices))


    def forward(self, input):
        '''

        :param input: B,F,
        :return: n_feat*(n_feat-1)//2
        '''

        x = input[:,self.x_idices,:]
        y = input[:,self.y_idices,:]

        r = torch.sum(x*y,dim=2)
        return r

class FFMP(Module):
    def __init__(self,n_feat,n_dim):
        super(FFMP,self).__init__()
        self.n_feat = n_feat
        self.n_dim = n_dim
        # generate idices
        x_idices = []
        y_idices = []
        for i in range(0,self.n_feat):
            for j in range(i,self.n_feat):
                x_idices.append(i*self.n_feat+j)
        for j in range(0,self.n_feat):
            for i in range(j,self.n_feat):
                y_idices.append(i*self.n_feat+j)
        x_idices = torch.LongTensor(x_idices)
        y_idices = torch.LongTensor(y_idices)
        self.register_buffer("x_idices",x_idices)
        self.register_buffer("y_idices", y_idices)
        print(self.x_idices)
        print(len(self.y_idices))
        self.out_dim = (self.n_dim*3+2)

    def forward(self, input):
        '''

        :param input: B,F,
        :return: n_feat*(n_feat-1)//2
        '''

        x = input[:,self.x_idices,:]
        y = input[:,self.y_idices,:]

        s = x+y
        d = x-y
        p = x*y
        ip = torch.sum(x*y,dim=2,keepdim=True)
        dist = torch.norm(x-y,2,dim=2,keepdim=True)
        r = torch.cat([s,d,p,ip,dist],dim=2)
        return r


if __name__=="__main__":
    # from torch.autograd import Variable
    # x = torch.Tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    # B, F, D = x.size()
    # x = Variable(x)
    # fm = FM(2)
    # z = fm(x)
    # print(z)
    # ffm = FFMSimple(2,2)
    # z = ffm(x)
    # print(z)
    x = torch.FloatTensor([[1,0,1],[-1,-1,1]])
    y = torch.FloatTensor([[1,2,-1],[-1,2,-3]])
    z = out_product(x,y)
    print(z)
    #ffm = FFM(4)
