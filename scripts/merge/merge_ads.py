adFeature_fn_A = "data/A/adFeature.csv"
adFeature_fn_F = "data/F/adFeature.csv"
adFeature_fn_M = "data/M_ff/adFeature.csv"

ads = dict()
with open(adFeature_fn_A) as f:
    header = f.readline()
    for line in f:
        records = line.strip().split(',')
        aid = int(records[0])
        ads[aid] = line

with open(adFeature_fn_F) as f:
    header = f.readline()
    for line in f:
        records = line.strip().split(',')
        aid = int(records[0])
        if aid in ads:
            print(aid)
            print(line)
        ads[aid] = line

with open(adFeature_fn_M,'w') as f:
    f.write(header)
    for line in ads.values():
        f.write(line)

