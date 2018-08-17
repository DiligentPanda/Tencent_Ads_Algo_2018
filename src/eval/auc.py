from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import traceback

def cal_auc(y_score,y_true):
    auc = roc_auc_score(y_true,y_score)
    return auc

def cal_avg_auc(pred,gt,use_file=True):
    if use_file:
        pred_df = pd.read_csv(pred)
        gt_df = pd.read_csv(gt)
    else:
        pred_df = pred
        gt_df = gt
    pred_groups = pred_df.groupby("aid")
    gt_groups = gt_df.groupby("aid")
    names = []
    aucs = []
    for name,gt_group in gt_groups:
        pred_group = pred_groups.get_group(name)
        pred_group.sort_values(by="uid")
        gt_group.sort_values(by="uid")
        flag = np.all(pred_group["uid"].values == gt_group["uid"].values)
        if not flag:
            raise Exception("uid mis-match")
        y_true = gt_group["label"]
        y_score = pred_group["score"]
        try:
            auc = cal_auc(y_score,y_true)
        except:
            print("auc fail")
            traceback.print_exc()
            auc = -10000
        aucs.append(auc)
        names.append(name)
    avg_auc = sum(aucs)/len(aucs)
    aucs = {n:a for n,a in zip(names,aucs)}
    return avg_auc,aucs


if __name__=="__main__":
    import numpy as np
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = roc_auc_score(y_true, y_scores)
    print(auc)
