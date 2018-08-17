train_fn_A = "data/A/train.csv"
train_fn_F = "data/F/train.csv"
train_fn_M = "data/M_ff/train.csv"

import shutil


shutil.copyfile(train_fn_F,train_fn_M)

fw = open(train_fn_M,'a')
with open(train_fn_A) as f:
    f.readline()
    for line in f:
        fw.write(line)

fw.close()
