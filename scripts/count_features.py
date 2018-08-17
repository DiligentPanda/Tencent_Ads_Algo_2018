user_fn = "data/M_ff/userFeature_mapped.data"
feat_info_fn = "data/M_ff/user_feat_infos.txt"
train_fn = "data/M_ff/train.csv"
features_mapping_fn = "data/M_ff/features_mapping.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping_fn,'rb') as f:
    feature_mappings = pkl.load(f)

from data_tool.feature import FeatureInfo
from data_tool.managers import Users

feat_names = [
  "age",
  "gender",
  "marriageStatus",
  "education",
  "consumptionAbility",
  "LBS",
  "interest1",
  "interest2",
  "interest3",
  "interest4",
  "interest5",
  "kw1",
  "kw2",
  "kw3",
  "topic1",
  "topic2",
  "topic3",
  "appIdInstall",
  "appIdAction",
  "ct",
  "os",
  "carrier",
  "house",
]

users = Users()
users.read(user_fn)

feat_infos = {name:FeatureInfo() for name in feat_names}

for k,v in feat_infos.items():
    v.construct(k,len(feature_mappings[k].mapping))

with open(train_fn) as f:
    f.readline()
    for n,line in enumerate(f):
        aid,uid,label = line.strip().split(',')
        uid, user = users.get(int(uid))
        assert len(user) == len(feat_names)
        for idx,feat_name in enumerate(feat_names):
            feat_infos[feat_name].ctr[user[idx]]+=1
        if (n+1)%100000==0:
            print(n+1)

with open(feat_info_fn,'w') as f:
    for k,v in feat_infos.items():
        f.write(v.to_str())

