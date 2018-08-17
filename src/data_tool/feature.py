import numpy as np
import logging
import warnings

class FeatureInfo:
    def __init__(self):
        self.name = None
        # include empty
        self.ma = None
        # note the second last is for empty, the last is for special value
        # counter
        self.ctr = None
        # counter for positive label
        self.mapper = None

    @property
    def empty_val(self):
        # it is empty val
        return self.ma

    @property
    def special_val(self):
        return self.ma+1

    @property
    def n_val(self):
        return len(self.ctr)

    def construct(self,name,ma):
        warnings.warn("This method assume empty val is not included in ma by default."
                      " It is not consistent with other function.")
        self.name = name
        self.ma = ma
        # todo we could use dict as ctr
        self.ctr = np.zeros(shape=(self.ma+2,),dtype=np.int32)
        self.mapper = None

    def __str__(self):
        string = "{name}|{ma}|{ctr}".format(
            name=self.name,
            ma=self.ma,
            ctr=" ".join([str(i) for i in self.ctr]),
        )
        return string

    __repr__ = __str__

    def to_str(self):
        string = "{}\n".format(self.__str__())
        return string

    def from_str(self,string,special_val=True):
        assert special_val==True
        string = string.strip()
        records = string.split("|")
        assert len(records)==3
        self.name = records[0]
        self.ma = int(records[1])
        self.ctr = np.fromstring(records[2].strip("]["), dtype=np.int32, sep=" ")
        self.mapper = None

    def construct_mapping(self):
        self.mapper = np.arange(0,len(self.ctr),dtype=np.int)
        logging.info("{} constructs its mapping, {} possible values.".format(self.name,self.n_val))

    def construct_filter(self, l_freq):
        '''

        :param lq: lowest frequency
        :return:
        '''
        self.filter = set()
        for i in range(self.ctr.size):
            if self.ctr[i]<l_freq:
                self.filter.add(i)
        # modify mapper and ctr
        self.ctr[self.special_val] = 0
        for i in self.filter:
            self.mapper[i] = self.special_val
            self.ctr[self.special_val] += self.ctr[i]
        logging.info("{} constructs its filter, {}/{} values of lower freq than {} are filterd, then mapped to special value {}".format(self.name,len(self.filter),self.n_val,l_freq,self.special_val))

    def map(self,idices):
        return np.unique(self.mapper[idices])

    def get_freqs(self,idices):
        '''
        Note !!! only mapped idices are supported.
        :param idices: mapped idices
        :return:
        '''
        warnings.warn("We should use mapped idices in get_freqs...")
        freqs = self.ctr[idices]
        return freqs


if __name__ == "__main__":
    fn = "data/A_shiyu/mapped_user_feature_infos.txt"
    from lib.debug.tools_v2 import load_feature_infos
    feature_infos = load_feature_infos(fn)
    feature_infos[5].construct_mapping()
    print(feature_infos[5].name,feature_infos[5].mapper.shape)
    print(feature_infos[5])
    #print(feature_infos)