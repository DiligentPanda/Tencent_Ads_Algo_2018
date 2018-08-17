'''
This scripts should run on raw user feature files.
'''

user_fn = "data/M_ff/userFeature.data"
new_user_fn = "data/M_ff/userFeature_mapped.data"
user_offsets_fn = "data/M_ff/userFeature_mapped_offsets.pkl"
features_mapping_fn = "data/M_ff/features_mapping.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping_fn,'rb') as f:
    features = pkl.load(f)

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

user_offsets = {}
offset = 0
with open(user_fn) as fr, open(new_user_fn,'w') as fw:
    for idx, line in enumerate(fr):
        records = line.strip().split("|")
        # the 0 is uid
        _, uid = records[0].split()
        user = {}
        for i in range(1, len(records)):
            words = records[i].split()
            keywords = words[0]
            values = [str(features[keywords].mapping[int(v)]) for v in words[1:]]
            #record = keywords+" "+" ".join(values)
            user[keywords] = values
        to_write = []
        to_write.append(uid)
        for feat_name in feat_names:
            if feat_name in user:
                to_write.append(" ".join(user[feat_name]))
            else:
                to_write.append(str(len(features[feat_name].mapping)))
        to_write = "|".join(to_write)+"\n"
        fw.write(to_write)
        user_offsets[int(uid)] = offset
        offset += len(to_write)
        if (idx + 1) % 100000 == 0:
            print(idx + 1)

with open(user_offsets_fn, 'wb') as f:
    pkl.dump(user_offsets,f)