import torch
import torch.nn as nn
import torch.nn.functional as F
from module.embedding_atten_v2 import EmbeddingAtten_v2
from module.FM import FFM,FM
from torch.nn import Module
import logging
from loss.focal_loss import bce_focal_loss
from loss.hinge_loss import hinge_loss
from module.dice import Dice

class ModelDINFFM_v3_r(Module):
    def __init__(self, n_out, u_embedding_feat_infos, u_one_hot_feat_infos,
                 a_embedding_feat_infos, a_one_hot_feat_infos, r_embedding_feat_infos, embedding_cfgs,
                 loss_cfg):
        '''

        :param n_out: the number of output of network, shoule be 1.
        :param u_embedding_feat_infos:
        :param u_one_hot_feat_infos:
        :param a_embedding_feat_infos:
        :param a_one_hot_feat_infos:
        :param r_embedding_feat_infos:
        :param embedding_cfgs:
        :param loss_cfg:
        '''

        super(ModelDINFFM_v3_r,self).__init__()
        assert len(u_one_hot_feat_infos) == 0
        assert len(a_one_hot_feat_infos) == 0

        self.u_embedding_feat_infos = u_embedding_feat_infos
        self.a_embedding_feat_infos = a_embedding_feat_infos
        # r_embedding_feat_infos is for history
        self.r_embedding_feat_infos = r_embedding_feat_infos
        self.embedding_feat_infos = self.u_embedding_feat_infos + self.a_embedding_feat_infos + self.r_embedding_feat_infos
        self.embedding_feat_infos_dict = {info.name: info for idx, info in enumerate(self.embedding_feat_infos)}

        self.n_field = len(self.embedding_feat_infos)
        self.n_embed_dim = list(embedding_cfgs.values())[0]["dim"]*self.n_field

        self.embedding_cfgs = embedding_cfgs
        self.loss_cfg = loss_cfg
        self.n_out = n_out

        self.construct_embedders()
        self.construct_loss(self.loss_cfg)

        self.n_feat = len(self.embedding_feat_infos)
        assert (self.n_field == self.n_feat)
        self.n_total_dim = sum([self.get_embedder(info.name).dim for info in self.embedding_feat_infos])

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fm = FFM(self.n_feat)
        #self.fm = FFMSimple(self.n_feat,self.n_embed_dim)

        #self.w_fm = nn.Linear(self.n_feat*(self.n_feat+1)//2,self.n_out)
        #self.w_1ord = nn.Linear(self.n_feat*self.n_embed_dim,self.n_out)

        self.n_output_feat = 64

        self.linear1 = nn.Linear(self.n_feat*self.n_embed_dim+self.n_feat*(self.n_feat+1)//2, 128)
        self.linear2 = nn.Linear(128, self.n_output_feat)
        self.linear3 = nn.Linear(64,self.n_out)#bias=False)
        self.dice2 = Dice(dim=self.n_output_feat)
        self.dice1 = Dice(dim=128)
        self.bn0 = nn.BatchNorm1d(num_features=self.n_feat*self.n_embed_dim+self.n_feat*(self.n_feat+1)//2)

        #self.init_weights()

    def init_weights(self):
        init = nn.init.xavier_normal
        init(self.linear1.weight)
        init(self.linear2.weight)
        init(self.linear3.weight)
        init(self.w_fm.weight)
        init(self.w_1ord.weight)
        logging.info("model init its weights.")

    def construct_embedders(self):
        for name,feat_info in self.embedding_feat_infos_dict.items():
            if name in self.embedding_cfgs:
                fconfig = self.embedding_cfgs[name]
            else:
                fconfig = {"dim": self.n_embed_dim//self.n_field, "atten": False}
            embedder_name = "embedder_{}".format(name)
            embedder = EmbeddingAtten_v2(num_embeddings=feat_info.n_val,
                                      embedding_dim=fconfig["dim"],
                                      n_field=self.n_field, atten=fconfig["atten"],
                                      mode="sum", norm_type=2, max_norm=1,
                                      padding_idx=feat_info.empty_val)
            # we simply use default orthogonal initialization
            embedder.init_weight(embedder_name,None)
            self.__setattr__(embedder_name,embedder)
            logging.info("{} constructs its embedder".format(name))

    def construct_loss(self,loss_cfg):
        from functools import partial
        if loss_cfg["name"] == "bce_focal":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return bce_focal_loss(input,target,gamma=loss_cfg["gamma"],weight=weight,size_average=size_average,reduce=reduce)
            self.loss=loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"],loss_cfg["gamma"]))
        elif loss_cfg["name"] == "bce":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return F.binary_cross_entropy_with_logits(input,target,weight=weight,size_average=size_average)
            self.loss=loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"],loss_cfg["gamma"]))
        elif loss_cfg["name"] == "hinge":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return hinge_loss(input, target, weight=weight, size_average=size_average)
            self.loss = loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"], loss_cfg["gamma"]))
        else:
            raise NotImplementedError

    def get_embedder(self,name):
        embedder_name = "embedder_{}".format(name)
        return self.__getattr__(embedder_name)

    def embed_reg(self,features_s):
        embeded_features = {}
        for fname, feature in features_s.items():
            embedder = self.get_embedder(fname)
            #print(fname)
            #print(features_s)
            embeded_feature = embedder.get_raw_embedding(input=feature[0])
            embeded_features[fname] = embeded_feature
        return embeded_features


    def embed(self,features_s,ref=None):
        '''

        :param features_s: it should be a dict!
        :return:
        '''
        embeded_features = {}
        for fname,feature in features_s.items():
            embedder = self.get_embedder(fname)
            # print(feature)
            if fname in self.embedding_cfgs:
                fconfig = self.embedding_cfgs[fname]
            else:
                fconfig = {"dim": self.n_embed_dim//self.n_field, "atten": False}
            if fconfig["atten"]:
                assert ref is not None
                embeded_feature = embedder(input=feature[0], offsets=feature[1], ref=ref)
            else:
                embeded_feature = embedder(input=feature[0], offsets=feature[1])
            embeded_features[fname] = embeded_feature
        return embeded_features

    def get_nonembedding_param_reg_term(self,lamb1=0.001,lamb2=0.001):
        norm = 0
        params =[self.linear1.weight,self.linear2.weight,self.linear3.weight,self.w_1ord.weight,self.w_fm.weight]
        for param in params:
            norm += (torch.norm(param,p=1) *lamb1 if lamb1 else 0) + (torch.norm(param,p=2)*lamb2 if lamb2 else 0)
        return norm

    def forward(self, samples):
        # features in sparse representation (feature values, offsets for each sample, ...), see Embedding class in pytorch.
        embedding_features_s = samples["embedding_features"]

        # obtain reference vector for attention, here it is the embedding of aid.
        ad_ref = self.embed({"aid": embedding_features_s["aid"]}, None)["aid"]

        embedding_features_d = self.embed(embedding_features_s, ad_ref)

        embedding_features = torch.cat(list(embedding_features_d.values()), dim=1)

        B, L = embedding_features.size()

        z = embedding_features

        # FFM
        r = self.fm(z.view(B, self.n_feat*self.n_field, self.n_embed_dim//self.n_field))
        #r = self.w_fm(r)

        x = torch.cat([embedding_features,r],dim=1)
        x = self.bn0(x)
        d = self.linear1(x)
        d = self.dice1(d)
        # d = self.bn1(d)
        # d = self.relu(d)
        d = self.linear2(d)
        d = self.dice2(d)
        # d = self.bn2(d)
        # d = self.relu(d)
        s = self.linear3(d)

        # d = self.tanh(d)
        # f1 = self.tanh(f1)
        # r = self.tanh(r)

        # s = torch.sum(r,dim=1)+d.view(-1)
        # s = d + f1 + r
        # print(d[:10],f1[:10],r[:10])
        s = s.view(-1)

        target = samples["labels"]
        target_weight = samples["label_weights"]
        if target is None:
            return None, s, None, None, d
        l = self.loss(s,target,target_weight,size_average=False)


        # average loss manually
        model_loss = l / samples["size"]

        final_loss = model_loss

        return final_loss, s, model_loss, 0, d

    def get_train_policy(self):
        # params_group1 = []
        # params_group2 = []
        # for name,params in self.named_parameters():
        #     if name.find("embedder")!=-1:
        #         params_group1.append(params)
        #         print("{} are in group 1 trained with lr x0.1".format(name))
        #     else:
        #         params_group2.append(params)
        #         print("{} are in group 1 trained with lr x1".format(name))
        #
        #
        #
        # params = [
        #     {'params': params_group1, "lr_mult": 0.1, "decay_mult": 1},
        #     {'params': params_group2, "lr_mult": 1, "decay_mult": 1},
        # ]

        params = [{'params': self.parameters(),"lr_mult": 1, "decay_mult": 1},]
        return params

    def get_embedder_params(self):
        params_group = []
        for name,params in self.named_parameters():
            if name.find("embedder.weight") != -1:
                 params_group.append(params)
                 logging.info("{} is embedder param".format(name))

        params = [{'params': params_group, "lr_mult": 1, "decay_mult": 1}, ]
        return params

    def get_other_params(self):
        params_group = []
        for name, params in self.named_parameters():
            if name.find("embedder.weight") == -1:
                params_group.append(params)
                logging.info("{} is other param".format(name))
        params = [{'params': params_group, "lr_mult": 1, "decay_mult": 1}, ]
        return params
