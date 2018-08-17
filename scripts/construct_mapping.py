'''
This scripts should run on raw user feature files.
'''

user_fn = "data/M_ff/userFeature.data"
feature_mapping_fn = "data/M_ff/features_mapping.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

features = {}

with open(user_fn) as f:
    for idx,line in enumerate(f):
        records = line.strip().split("|")
        # the 0 is uid
        for i in range(1,len(records)):
            words = records[i].split()
            keywords = words[0]
            values = [int(v) for v in words[1:]]
            if keywords not in features:
                features[keywords] = Feature(keywords)
            features[keywords].values.update(values)
        if (idx+1)%100000==0:
            print(idx+1)

for feature in features.values():
    feature.values = sorted(list(feature.values))
    feature.mapping = {val:idx for idx,val in enumerate(feature.values)}

with open(feature_mapping_fn,'wb') as f:
    pkl.dump(features,f)


