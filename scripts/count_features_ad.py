ad_fn = "data/M_ff/adFeature_mapped.data"
feat_info_fn = "data/M_ff/ad_feat_infos.txt"
train_fn = "data/M_ff/train.csv"
features_mapping_fn = "data/M_ff/features_mapping_ad.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping_fn,'rb') as f:
    feature_mappings = pkl.load(f)

from data_tool.feature import FeatureInfo
from data_tool.managers import Ads

feat_names = [
  "aid",
  "advertiserId",
  "campaignId",
  "creativeId",
  "creativeSize",
  "adCategoryId",
  "productId",
  "productType",
]

ads = Ads()
ads.read(ad_fn)

feat_infos = {name:FeatureInfo() for name in feat_names}

for k,v in feat_infos.items():
    v.construct(k,len(feature_mappings[k].mapping))

with open(train_fn) as f:
    f.readline()
    for idx,line in enumerate(f):
        aid,uid,label = line.strip().split(',')
        aid, ad = ads.get(int(aid))
        assert len(ad) == len(feat_names)
        for i,feat_name in enumerate(feat_names):
            feat_infos[feat_name].ctr[ad[i]]+=1
        if (idx+1)%100000==0:
            print(idx+1)


with open(feat_info_fn,'w') as f:
    for k,v in feat_infos.items():
        f.write(v.to_str())

