train_fn = "data/M_ff/train.csv"
test_fn = "data/M_ff/test2.csv"
train_his_fn = "data/M_ff/train_his.csv"
test_his_fn = "data/M_ff/test2_his.csv"
ads_fn = "data/M_ff/adFeature_mapped.data"

ads = {}
ads_feats = {}
others = {}
with open(ads_fn) as f:
    f.readline()
    for idx,line in enumerate(f):
        records = line.strip().split(',')
        aid = records[0]
        for field_idx,r in enumerate(records[1:]):
            val = "{}_{}".format(field_idx,r)
            if val not in others:
                others[val] = str(len(others))
        ads[aid] = str(idx)
        ads_feats[ads[aid]] = records[1:]
print(ads)
print(others)


import pickle

n_ads = 1005
n_others = len(others)
assert len(ads)==n_ads
#todo: no need to +1, but we just let it go. a tiny bug
empty_val = str(n_ads+1)
empty_val_others = str(n_others+1)

print(len(others))

users = {}
with open(train_fn) as f:
    f.readline()
    for idx,line in enumerate(f):
        aid,uid,label = line.strip().split(',')
        aid = aid
        uid = uid
        label = int(label)
        if uid not in users:
            users[uid] = [[],[]]
        if label==1:
            users[uid][0].append(ads[aid])
        else:
            users[uid][1].append(ads[aid])
        if (idx+1)%100000==0:
            print(idx+1)

#print(users)

import copy

with open(train_fn) as fr, open(train_his_fn,'w') as fw:
    fr.readline()
    fw.write("pos_his,neg_his,pos_feats,neg_feats\n")
    for idx,line in enumerate(fr):
        aid, uid, label = line.strip().split(',')
        aid = aid
        uid = uid
        if uid in users:
            his = copy.deepcopy(users[uid])
        else:
            his = [[], []]
        if ads[aid] in his[0]:
            his[0].remove(ads[aid])
        if ads[aid] in his[1]:
            his[1].remove(ads[aid])
        pos_feats = set()
        neg_feats = set()
        for id in his[0]:
            ad_feats = ads_feats[id]
            pos_feats.update([others["{}_{}".format(field_idx,r)] for field_idx,r in enumerate(ad_feats)])
        for id in his[1]:
            ad_feats = ads_feats[id]
            neg_feats.update([others["{}_{}".format(field_idx,r)] for field_idx,r in enumerate(ad_feats)])
        pos_feats = list(pos_feats)
        neg_feats = list(neg_feats)
        if len(his[0]) == 0:
            his[0].append(empty_val)
        if len(his[1]) == 0:
            his[1].append(empty_val)
        if len(pos_feats)==0:
            pos_feats.append(empty_val_others)
        if len(neg_feats)==0:
            neg_feats.append(empty_val_others)
        fw.write(" ".join(his[0]) + "," + " ".join(his[1])+","+" ".join(pos_feats)+","+" ".join(neg_feats) + "\n")

# with open(valid_fn) as fr, open(valid_his_fn,'w') as fw:
#     fr.readline()
#     fw.write("pos_his,neg_his\n")
#     for idx,line in enumerate(fr):
#         aid, uid, label = line.strip().split(',')
#         aid = aid
#         uid = uid
#         if uid in users:
#             his = copy.deepcopy(users[uid])
#             #print(his)
#         else:
#             his = [[],[]]
#         if ads[aid] in his[0]:
#             his[0].remove(ads[aid])
#         if ads[aid] in his[1]:
#             his[1].remove(ads[aid])
#         if len(his[0])==0:
#             his[0].append(empty_val)
#         if len(his[1])==0:
#             his[1].append(empty_val)
#         fw.write(" ".join(his[0])+","+" ".join(his[1])+"\n")

with open(test_fn) as fr, open(test_his_fn,'w') as fw:
    fr.readline()
    fw.write("pos_his,neg_his,pos_feats,neg_feats\n")
    for idx,line in enumerate(fr):
        aid, uid = line.strip().split(',')
        aid = aid
        uid = uid
        if uid in users:
            his = copy.deepcopy(users[uid])
        else:
            his = [[], []]
        if ads[aid] in his[0]:
            his[0].remove(ads[aid])
        if ads[aid] in his[1]:
            his[1].remove(ads[aid])
        pos_feats = set()
        neg_feats = set()
        for id in his[0]:
            ad_feats = ads_feats[id]
            pos_feats.update([others["{}_{}".format(field_idx,r)] for field_idx,r in enumerate(ad_feats)])
        for id in his[1]:
            ad_feats = ads_feats[id]
            neg_feats.update([others["{}_{}".format(field_idx,r)] for field_idx,r in enumerate(ad_feats)])
        pos_feats = list(pos_feats)
        neg_feats = list(neg_feats)
        if len(his[0]) == 0:
            his[0].append(empty_val)
        if len(his[1]) == 0:
            his[1].append(empty_val)
        if len(pos_feats)==0:
            pos_feats.append(empty_val_others)
        if len(neg_feats)==0:
            neg_feats.append(empty_val_others)
        fw.write(" ".join(his[0]) + "," + " ".join(his[1])+","+" ".join(pos_feats)+","+" ".join(neg_feats) + "\n")

print(n_ads,n_others)