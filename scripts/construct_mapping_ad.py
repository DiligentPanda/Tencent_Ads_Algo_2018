'''
This scripts should run on raw user feature files.
'''

user_fn = "data/M_ff/adFeature.csv"
feature_mapping_fn = "data/M_ff/features_mapping_ad.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

features = {}

with open(user_fn) as f:
    line = f.readline()
    feat_names = line.strip().split(",")
    for feat_name in feat_names:
        features[feat_name] = Feature(feat_name)
    for idx,line in enumerate(f):
        records = line.strip().split(",")
        for i,feat_name in enumerate(feat_names):
            features[feat_name].values.add(int(records[i]))
        if (idx+1)%100000==0:
            print(idx+1)

for feature in features.values():
    feature.values = sorted(list(feature.values))
    feature.mapping = {val:idx for idx,val in enumerate(feature.values)}

print(features["aid"].mapping)

with open(feature_mapping_fn,'wb') as f:
    pkl.dump(features,f)


