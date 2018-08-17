'''
This scripts should run on raw user feature files.
'''

ad_fn = "data/M_ff/adFeature.csv"
new_ad_fn = "data/M_ff/adFeature_mapped.data"
features_mapping_fn = "data/M_ff/features_mapping_ad.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping_fn,'rb') as f:
    features = pkl.load(f)

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

with open(ad_fn) as fr, open(new_ad_fn,'w') as fw:
    line = fr.readline()
    fw.write("raid,"+line)
    for idx, line in enumerate(fr):
        records = line.strip().split(",")
        to_write = [records[0]]
        for i,feat_name in enumerate(feat_names):
            to_write.append(str(features[feat_name].mapping[int(records[i])]))
        to_write = ",".join(to_write)+"\n"
        fw.write(to_write)
        if (idx + 1) % 100000 == 0:
            print(idx + 1)