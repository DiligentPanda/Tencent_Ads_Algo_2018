class RunningValue:
    def __init__(self,window_size=1000):
        self.values = []
        self.window_size = window_size

    @property
    def sum(self):
        return sum(self.values)

    def update(self,v):
        if len(self.values)<self.window_size:
            self.values.append(v)
        else:
            self.values.pop(0)
            self.values.append(v)

    @property
    def mean(self):
        return sum(self.values)/len(self.values)

    def reset(self):
        self.values = []


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.ctr = 0

    def update(self,v,n=1):
        self.sum += v*n
        self.ctr += n

    @property
    def mean(self):
        return self.sum/self.ctr

    def reset(self):
        self.sum = 0
        self.ctr = 0

class SumMeter:
    def __init__(self):
        self.sum = 0
        self.ctr = 0

    def update(self,v,n=1):
        self.sum += v
        self.ctr += n

    @property
    def mean(self):
        return self.sum/self.ctr

    def reset(self):
        self.sum = 0
        self.ctr = 0


from eval.auc import cal_avg_auc
import pandas as pd

class AUCMeter:
    def __init__(self):
        self.gt = []
        self.pred = []

    def update(self,pred,gt):
        self.gt.extend(gt)
        self.pred.extend(pred)

    @property
    def auc(self):
        pred = pd.DataFrame(self.pred, columns=['aid','uid','score'])
        gt = pd.DataFrame(self.gt, columns=['aid','uid','label'])
        auc,_ = cal_avg_auc(pred,gt,use_file=False)
        return auc

    def reset(self):
        self.gt = []
        self.pred = []


if __name__=="__main__":
    rv = RunningValue(3)
    for i in [3,2,4,1,5]:
        rv.update(i)
        print(rv.mean)
    sm = SumMeter()
    for i in [3,2,4,1,5]:
        sm.update(i,2)
        print(sm.mean)