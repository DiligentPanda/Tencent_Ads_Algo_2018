'''
1.
2018.04.29
jh
This file is copied from dataset_baby.py
This time, however, we will add ads features.
2.
2018.04.30
This file is copied from dataset_infant.py
one_hot encoding would be adjusted to dict.
3.
2018.05.06
copied from dataset_child.py
We add infor related to regularization in get.
reg_features and reg_weight.
Currently we only penalize user features.
'''


import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import logging
from collections import Counter
import traceback
# from ipdb import set_trace
import traceback
import random

class DatasetYouth(Dataset):
    class CodingMethod:
        DROP = 0
        ONE_HOT = 1
        EMBEDDING = 2

    def shuffle(self):
        np.random.shuffle(self.original_indices)
        logging.info("dataset is shuffled...")

    @property
    def original_len(self):
        return len(self.original_indices)

    def slice(self,s=0,e=None):
        self.indices = self.original_indices[s:e]
        return self

    def __init__(self, users, u_feat_infos, ads, a_feat_infos, rfeats, r_feat_infos, data_list, u_enc_method, a_enc_method, r_enc_method, reg, pos_weight=1, has_label = True):
        '''

        :param users:
        :param u_feat_infos:
        :param ads:
        :param a_feat_infos:
        :param rfeats: for history features
        :param r_feat_infos: feature information for history features.
        :param data_list:
        :param u_enc_method:
        :param a_enc_method:
        :param r_enc_method:
        :param reg: configuration of regularization
        :param pos_weight: loss weight for positive samples
        :param has_label: whether this dataset has labels
        '''
        super(DatasetYouth,self).__init__()

        self.pos_weight = pos_weight
        self.reg = reg

        self.users = users
        self.u_feat_infos = u_feat_infos
        self.u_feat_infos_dict = {info.name:info for info in u_feat_infos}

        self.ads = ads
        self.a_feat_infos = a_feat_infos
        self.a_feat_infos_dict = {info.name:info for info in a_feat_infos}

        self.rfeats = rfeats
        self.r_feat_infos = r_feat_infos
        self.r_feat_infos_dict = {info.name:info for info in r_feat_infos}

        self.has_label = has_label

        self.n_u_feat = len(self.u_feat_infos)
        self.n_a_feat = len(self.a_feat_infos)
        self.n_r_feat = len(self.r_feat_infos)

        self.u_feat_name = [fi.name for fi in self.u_feat_infos]
        self.a_feat_name = [fi.name for fi in self.a_feat_infos]
        self.r_feat_name = [fi.name for fi in self.r_feat_infos]

        self.data_list = data_list
        self.rfeats = rfeats
        assert len(self.data_list)==len(self.rfeats)

        # this is used for get a slice for dataset.
        self.original_indices = np.arange(len(self.data_list))
        self.indices = self.original_indices

        '''
        # this sould not be used for scalability.
        self.one_hot_list = ['age','gender','marriageStatus',
                             'education','consumptionAbility',
                             'ct','os','carrier','house']
        self.embedding_list = ['LBS',
                               'interest1','interest2','interest3','interest4','interest5',
                               'kw1','kw2','kw3',
                               'topic1','topic2','topic3',
                               'appIdInstall','appIdAction',
                               ]
        '''
        self.u_enc_method = u_enc_method
        self.a_enc_method = a_enc_method

        self.r_enc_method = {}
        for name in self.r_feat_name:
            if name in r_enc_method:
                self.r_enc_method[name] = r_enc_method[name]
            else:
                self.r_enc_method[name] = self.CodingMethod.DROP

        self.embedding_u_feat_infos = self._construct_type_u_feat_infos(self.CodingMethod.EMBEDDING)
        self.embedding_a_feat_infos = self._construct_type_a_feat_infos(self.CodingMethod.EMBEDDING)
        self.embedding_r_feat_infos = self._construct_type_r_feat_infos(self.CodingMethod.EMBEDDING)

        self.embedding_feat_infos = self.embedding_u_feat_infos+self.embedding_a_feat_infos + self.embedding_r_feat_infos

        self.one_hot_u_feat_infos = self._construct_type_u_feat_infos(self.CodingMethod.ONE_HOT)
        self.one_hot_a_feat_infos = self._construct_type_a_feat_infos(self.CodingMethod.ONE_HOT)
        self.one_hot_feat_infos = self.one_hot_u_feat_infos + self.one_hot_a_feat_infos

    def _construct_type_u_feat_infos(self,encoding_type):
        embedding_feat_infos = []
        for idx,fname in enumerate(self.u_feat_name):
            if self.u_enc_method[fname] == encoding_type:
                embedding_feat_infos.append(self.u_feat_infos[idx])
        return embedding_feat_infos

    def _construct_type_a_feat_infos(self,encoding_type):
        embedding_feat_infos = []
        for idx,fname in enumerate(self.a_feat_name):
            if self.a_enc_method[fname] == encoding_type:
                embedding_feat_infos.append(self.a_feat_infos[idx])
        return embedding_feat_infos

    def _construct_type_r_feat_infos(self,encoding_type):
        embedding_feat_infos = []
        for idx,fname in enumerate(self.r_feat_name):
            if self.r_enc_method[fname] == encoding_type:
                embedding_feat_infos.append(self.r_feat_infos[idx])
        return embedding_feat_infos


    def get_collate_func(self):
        '''
        Note currently ad feature and user feature are returned together.
        Label are only return as 0 or 1.
        :return:
        '''
        # todo test it
        def _collate(batch):
            uids = []
            aids = []

            u_one_hot_features = {}
            for fname in self.u_feat_name:
                if self.u_enc_method[fname] == self.CodingMethod.ONE_HOT:
                    u_one_hot_features[fname] = []

            u_embedding_features = {}
            for fname in self.u_feat_name:
                if self.u_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    # input, offset, ctr (counting offset): see torch.nn.EmbeddingBag
                    u_embedding_features[fname] = [[],[],0]

            a_one_hot_features = {}
            for fname in self.a_feat_name:
                if self.a_enc_method[fname] == self.CodingMethod.ONE_HOT:
                    a_one_hot_features[fname] = []

            a_embedding_features = {}
            for fname in self.a_feat_name:
                if self.a_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    # input, offset, ctr (counting offset): see torch.nn.EmbeddingBag
                    a_embedding_features[fname] = [[], [], 0]

            r_embedding_features = {}
            for fname in self.r_feat_name:
                if self.r_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    r_embedding_features[fname] = [[],[],0]

            labels = []
            label_weights = []

            for sid, sample in enumerate(batch):
                user, ad, rfeat, label, label_weight, uid, aid = sample

                uids.append(uid)
                aids.append(aid)

                labels.append(label)
                label_weights.append(label_weight)

                for fid, fname in enumerate(self.u_feat_name):
                    #print(fid, fname,len(user))
                    if self.u_enc_method[fname] == self.CodingMethod.ONE_HOT:
                        u_one_hot_features[fname].append(user[fid])
                    elif self.u_enc_method[fname] == self.CodingMethod.EMBEDDING:
                        # feature value idices
                        u_embedding_features[fname][0].append(user[fid])
                        # offset
                        u_embedding_features[fname][1].append(u_embedding_features[fname][2])
                        # accumulate the length
                        u_embedding_features[fname][2] += len(u_embedding_features[fname][0][-1])
                    elif self.u_enc_method[fname] == self.CodingMethod.DROP:
                        pass
                    else:
                        raise NotImplementedError

                for fid, fname in enumerate(self.a_feat_name):
                    #print(fid, fname,len(user))
                    if self.a_enc_method[fname] == self.CodingMethod.ONE_HOT:
                        a_one_hot_features[fname].append(ad[fid])
                    elif self.a_enc_method[fname] == self.CodingMethod.EMBEDDING:
                        # feature value idices
                        a_embedding_features[fname][0].append(ad[fid])
                        # offset
                        a_embedding_features[fname][1].append(a_embedding_features[fname][2])
                        # accumulate the length
                        a_embedding_features[fname][2] += len(a_embedding_features[fname][0][-1])
                    elif self.a_enc_method[fname] == self.CodingMethod.DROP:
                        pass
                    else:
                        raise NotImplementedError

                for fid,fname in enumerate(self.r_feat_name):
                    if self.r_enc_method[fname] == self.CodingMethod.EMBEDDING:
                        r_embedding_features[fname][0].append(rfeat[fid])
                        r_embedding_features[fname][1].append(r_embedding_features[fname][2])
                        r_embedding_features[fname][2] += len(rfeat[fid])

            if self.has_label:
                labels = torch.FloatTensor(np.array(labels))
                label_weights = torch.FloatTensor(np.array(label_weights))
            else:
                labels = None
                label_weights = None

            for fname in self.u_feat_name:
                if self.u_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    u_embedding_features[fname][0] = torch.LongTensor(np.concatenate(u_embedding_features[fname][0]))
                    u_embedding_features[fname][1] = torch.LongTensor(np.array(u_embedding_features[fname][1]))
                elif self.u_enc_method[fname] == self.CodingMethod.ONE_HOT:
                    u_one_hot_features[fname] = torch.FloatTensor(np.vstack(u_one_hot_features[fname]))
                elif self.u_enc_method[fname] == self.CodingMethod.DROP:
                    pass
                else:
                    raise NotImplementedError

            for fname in self.a_feat_name:
                if self.a_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    a_embedding_features[fname][0] = torch.LongTensor(np.concatenate(a_embedding_features[fname][0]))
                    a_embedding_features[fname][1] = torch.LongTensor(np.array(a_embedding_features[fname][1]))
                elif self.a_enc_method[fname] == self.CodingMethod.ONE_HOT:
                    a_one_hot_features[fname] = torch.FloatTensor(np.vstack(a_one_hot_features[fname]))
                elif self.a_enc_method[fname] == self.CodingMethod.DROP:
                    pass
                else:
                    raise NotImplementedError

            for fname in self.r_feat_name:
                if self.r_enc_method[fname] == self.CodingMethod.EMBEDDING:
                    r_embedding_features[fname][0] = torch.LongTensor(np.concatenate(r_embedding_features[fname][0]))
                    r_embedding_features[fname][1] = torch.LongTensor(np.array(r_embedding_features[fname][1]))

            one_hot_features = {}
            one_hot_features.update(u_one_hot_features)
            one_hot_features.update(a_one_hot_features)

            embedding_features = {}
            embedding_features.update(u_embedding_features)
            embedding_features.update(a_embedding_features)
            embedding_features.update(r_embedding_features)

            return {
                "one_hot_features": one_hot_features,
                "embedding_features": embedding_features,
                "labels": labels,
                "label_weights": label_weights,
                "size": len(batch),
                "uids": uids,
                "aids": aids,
            }
        return _collate

    def get_user(self,uid):
        try:
            uid, raw_feature = self.users.get(uid)
            # todo store intermediate results
            mapped_feature = [self.u_feat_infos[i].map(raw_feature[i]) for i in range(self.n_u_feat)]
            coded_feature = []
            for i in range(self.n_u_feat):
                if self.u_enc_method[self.u_feat_name[i]] == self.CodingMethod.ONE_HOT:
                    vec = np.zeros(self.u_feat_infos[i].n_val,dtype=np.float32)
                    vec[mapped_feature[i]] = 1
                    coded_feature.append(vec)
                elif self.u_enc_method[self.u_feat_name[i]] == self.CodingMethod.EMBEDDING:
                    # directly pass them to embedding would be fine
                    coded_feature.append(mapped_feature[i])
                elif self.u_enc_method[self.u_feat_name[i]] == self.CodingMethod.DROP:
                    coded_feature.append(None)
                else:
                    raise NotImplementedError
        except Exception as e:
            print("error",uid)
            #print(raw_feature)
            traceback.print_exc()
            #set_trace()
        return uid, coded_feature

    def get_ad(self,aid):
        aid, raw_feature = self.ads.get(aid)
        mapped_feature = [self.a_feat_infos[i].map(raw_feature[i]) for i in range(self.n_a_feat)]
        coded_feature = []
        for i in range(self.n_a_feat):
            if self.a_enc_method[self.a_feat_name[i]] == self.CodingMethod.ONE_HOT:
                vec = np.zeros(self.a_feat_infos[i].n_val, dtype=np.float32)
                vec[mapped_feature[i]] = 1
                coded_feature.append(vec)
            elif self.a_enc_method[self.a_feat_name[i]] == self.CodingMethod.EMBEDDING:
                coded_feature.append(mapped_feature[i])
            elif self.a_enc_method[self.a_feat_name[i]] == self.CodingMethod.DROP:
                coded_feature.append(None)
            else:
                raise NotImplementedError

        return aid, coded_feature

    def get_rfeat(self,idx):
        feature = []
        for i in range(self.n_r_feat):
            if self.r_enc_method[self.r_feat_name[i]] == self.CodingMethod.EMBEDDING:
                feature.append(self.rfeats[idx][i])
            else:
                feature.append(None)
        # no need to do any thing
        return feature

    def __getitem__(self, idx):
        idx = self.indices[idx]
        if self.has_label:
            t = self.data_list[idx]
            aid = t[0]
            uid = t[1]
            rel = t[2]
            uid, user = self.get_user(uid)
            aid, ad = self.get_ad(aid)
            rfeat = self.get_rfeat(idx)
            # convert ad to label, mask
            # mask(label_weight) are used for training only one classifier
            if rel==1:
                label = 1
                label_weight = self.pos_weight
            else:
                label = 0
                label_weight = 1
        else:
            # for valid as test
            t  = self.data_list[idx]
            aid = t[0]
            uid = t[1]
            uid, user = self.get_user(uid)
            aid, ad = self.get_ad(aid)
            rfeat = self.get_rfeat(idx)
            label = None
            label_weight = None
        return user, ad, rfeat, label, label_weight, uid, aid

    def __len__(self):
        return len(self.indices)

    def get_to_gpu_variable_func(self):
        def to_gpu_variable(samples,volatile=False):
            for k, v in samples["one_hot_features"].items():
                samples["one_hot_features"][k] = Variable(v, volatile=volatile)
            samples["labels"] = Variable(samples["labels"].cuda(),volatile=volatile) if samples["labels"] is not None else None
            samples["label_weights"] = Variable(samples["label_weights"].cuda(),volatile=volatile) if samples["label_weights"] is not None else None
            for k,v in samples["embedding_features"].items():
                samples["embedding_features"][k] = [Variable(v[0].cuda(),volatile=volatile),Variable(v[1].cuda(),volatile=volatile),v[2]]
            return samples
        return to_gpu_variable

if __name__=="__main__":
    user_fn = "data/userFeature_clean.data"
    attr_list_fn = "data0/user_attr_list.txt"
    ad_fn = "data/adFeature.csv"
    user_feat_fn = "data/user_feature_infos.txt"
    ad_feat_fn = "data/ad_feature_infos.txt"

    data_fn = "data/train.csv"

    config_fn = "exp/baby.yaml"

    from lib.tools import load_users_and_ads,load_data_list,read_config

    users,ads,u_feat_infos,a_feat_infos = load_users_and_ads(user_fn,attr_list_fn,ad_fn,user_feat_fn,ad_feat_fn)
    data_list = load_data_list(data_fn)
    config = read_config(config_fn)

    dataset = DatasetBaby(users,u_feat_infos,ads,a_feat_infos,data_list,config["feat"]["u_enc"])

    batch = []
    for i in range(10):
        batch.append(dataset[i])
    print(batch[0])

    collate = dataset.get_collate_func()
    one_hot_features,embedding_features, labels, label_weights = collate(batch)
    print(embedding_features)


