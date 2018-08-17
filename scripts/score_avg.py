import numpy as np
import pandas as pd
import os

output_fn = "submission.csv"
file_list_fn = "output/M_ff/test2_file_list.txt"

fns = {}
with open(file_list_fn) as f:
    for line in f:
        fn, weight = line.split()
        weight = float(weight)
        fns[fn] = weight

total_weight = sum(fns.values())

t_df = pd.DataFrame()
for idx,(fn,weight) in enumerate(fns.items()):
    df = pd.read_csv(fn)
    if idx==0:
        t_df["aid"] = df["aid"]
        t_df["uid"] = df["uid"]
    t_df["score_{}".format(idx)] = df["score"]*weight/total_weight

n=len(fns)
print("average over ",fns)

f_df = t_df
score_fields = ["score_{}".format(i) for i in range(n)]
f_df["score"] = f_df[score_fields].sum(axis=1)
f_df = f_df[["aid","uid","score"]]
f_df.round({"score":8})

f_df.to_csv(output_fn,float_format="%.8f",index=False)
