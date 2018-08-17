import torch
from torch.utils.data.sampler import Sampler
import numpy as np

class RadioSampler(Sampler):
    def __init__(self, data_source, p2n_radio=1, num=None):
        super(RadioSampler,self).__init__(data_source)
        self.p2n_radio = p2n_radio
        # manage data_source into two set
        self.pos_idices = []
        self.neg_idices = []
        for idx,data in enumerate(data_source.data_list):
            # label
            if data[2]==1:
                self.pos_idices.append(idx)
            else:
                self.neg_idices.append(idx)
        self.pos_idices = np.array(self.pos_idices)
        self.neg_idices = np.array(self.neg_idices)
        if num is not None:
            self.num = num
        else:
            self.num = len(data_source)

    def __iter__(self):
        n_pos = int(self.num * self.p2n_radio / (self.p2n_radio+1))
        n_neg = self.num - n_pos
        pos_idx_idices = np.random.randint(low=0,high=len(self.pos_idices),size=(n_pos,))
        neg_idx_idcies = np.random.randint(low=0,high=len(self.neg_idices),size=(n_neg,))
        pos_idices = self.pos_idices[pos_idx_idices]
        neg_idices = self.neg_idices[neg_idx_idcies]
        idices = np.concatenate((pos_idices,neg_idices),axis=0)
        np.random.shuffle(idices)
        return iter(idices)

    def __len__(self):
        return self.num

if __name__=="__main__":
    class TestDataset:
        def __init__(self):
            self.data_list = [[1,1,1],
                              [2,1,-1],
                              [2,1,-1],
                              [1,2,-1],
                              [2,3,-1],
                              [3,4,-1],
                              [2,5,1],
                              ]

        def __len__(self):
            return len(self.data_list)

    dataset = TestDataset()

    radio_sampler = RadioSampler(dataset,0.5,num=12)

    for i in range(5):
        print(list(iter(radio_sampler)))
