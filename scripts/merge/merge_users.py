userFeature_fn_A = "data/A/userFeature.data"
userFeature_fn_F = "data/F/userFeature.data"
userFeature_fn_M = "data/M_ff/userFeature.data"

users = dict()
with open(userFeature_fn_A) as f:
    for line in f:
        records = line.strip().split('|')
        uid = int(records[0].split()[1])
        users[uid] = line

with open(userFeature_fn_F) as f:
    for line in f:
        records = line.strip().split('|')
        uid = int(records[0].split()[1])
        if uid in users:
            if line!=users[uid]:
                print(users[uid])
                print(line)
        users[uid] = line

with open(userFeature_fn_M,'w') as f:
    for line in users.values():
        f.write(line)

