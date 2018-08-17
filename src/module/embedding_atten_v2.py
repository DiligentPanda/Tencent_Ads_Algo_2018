import torch
from torch.nn import Module, Embedding, Parameter, Linear, PReLU,Sigmoid
from module.tool import reduce,replicate
import numpy as np
import logging

class EmbeddingAtten_v2(Module):
    def __init__(self, num_embeddings, embedding_dim, n_field, atten=True, mode="sum",
                 padding_idx=None, max_norm=1,
                 norm_type=2, scale_grad_by_freq=False,
                 sparse=False):
        super(EmbeddingAtten_v2, self).__init__()

        self.mode = mode
        self.atten = atten
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n_field = n_field
        self.dim = n_field*embedding_dim
        self.padding_idx = padding_idx

        self.embedder = Embedding(num_embeddings, self.dim,
                                  padding_idx, max_norm=None,
                                  norm_type=None, scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)

        self.max_norm = max_norm
        self.norm_type = norm_type

        if atten:
            # for attention
            self.linear1 = Linear(3*self.dim,self.dim)
            self.linear2 = Linear(self.dim,1)
            # activation
            # todo Note in ali's paper there is a kind of activation called Dice
            # but it is too complicated, we use PReLU
            self.activation = PReLU()
            self.sigmoid = Sigmoid()



    def init_weight(self,name,weight_fp):
        if weight_fp:
            weight = np.load(weight_fp)
            self.embedder.weight = Parameter(torch.FloatTensor(weight))
            logging.info("load {}'s weight from {}".format(name,weight_fp))
        else:
            weight = self.embedder.weight.view(-1,self.embedding_dim)
            torch.nn.init.orthogonal(weight)
            if self.padding_idx is not None:
                self.embedder.weight.data[self.padding_idx].fill_(0)
            logging.info("init {}'s weight orthogonal".format(name))


    def get_raw_embedding(self, input):
        input = input.view(1, -1)
        # return 1, n_word, n_dim
        embedding = self.embedder(input)
        # print(embedding)
        size = embedding.size()
        # n_word, n_dim
        embedding = embedding.view(size[1], size[2])
        return embedding

    #@timethis
    def forward(self, input, offsets, ref=None):
        '''

        :param input:  a 1-dim tensor of indices
        :param offset: a 1-dim tensor of offsets
        :param ref: a 2-dim tensor of ref feats, typically the features of ads
        :return:
        '''
        assert (ref is None and not self.atten) or (ref is not None and self.atten)
        # add 1 dim for Embedding
        input = input.view(1,-1)
        # return 1, n_word, n_dim
        embedding = self.embedder(input)
        #print(embedding)
        size = embedding.size()
        # n_word, n_dim
        embedding = embedding.view(size[1],size[2])
        if self.atten:
            size = embedding.size()
            # replicate ref n_word, n_dim
            ref = replicate(ref,offsets,size[0])
            #print(ref)
            # calculate the attention
            #todo
            diff = ref-embedding
            feat_for_atten = torch.cat([embedding,diff,ref],dim=1)
            atten = self.linear1(feat_for_atten)
            atten = self.activation(atten)
            atten = self.linear2(atten)
            # n_word, 1
            atten = self.sigmoid(atten)
            # print(atten)
            embedding = embedding * atten
            #print(embedding)
        # n_sample, n_dim
        res = reduce(embedding,offsets,self.mode)
        # following lines constrain the max norm of embedding.
        size = res.size()
        # n_sample, n_field, n_dim//n_field
        res = res.view(size[0]*self.n_field,size[1]//self.n_field)
        renorm_res = torch.renorm(res,p=self.norm_type,dim=0,maxnorm=self.max_norm)
        renorm_res = renorm_res.contiguous()
        # res = F.normalize(res,p=self.norm_type,dim=2)*self.max_norm
        res = renorm_res.view(size[0],size[1])
        return res


if __name__=="__main__":
    from torch.autograd import Variable

    embedder = EmbeddingAtten_v2(4,3,2,atten=True,mode="sum",padding_idx=0)
    #embedder.init_weight("hello",None)
    input = Variable(torch.LongTensor([1,0,0,0,0,3,2]))
    offsets = Variable(torch.LongTensor([0,2,4]))
    ref = Variable(torch.Tensor([[1,0,1,0,0,1],[1,-1,-1,1,0,1],[-1,-1,0,0,1,1]]))
    res = embedder(input,offsets,ref)
    print(res)



