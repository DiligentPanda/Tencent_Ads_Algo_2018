with open("data/M_ff/train_his.csv") as f:
    line = f.readline().strip()
    headers = line.split(',')

feat_info_fn = "data/M_ff/his_feat_infos.txt"

from data_tool.feature import FeatureInfo

feat_infos = {name:FeatureInfo() for name in headers}

for k,v in feat_infos.items():
    if k in ["pos_his","neg_his"]:
        v.construct(k,1005)
    else:
        v.construct(k,2951)

with open(feat_info_fn,'w') as f:
    for k,v in feat_infos.items():
        f.write(v.to_str())