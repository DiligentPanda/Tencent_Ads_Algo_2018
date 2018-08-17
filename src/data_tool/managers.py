
class Users:
    def __init__(self):
        self.user_fn = None
        self.users = None
        self.uid2id = {}
        self.id2uid = []

    def read(self, user_fn):
        self.user_fn = user_fn
        with open(self.user_fn) as f:
            self.users = f.readlines()
            for idx,line in enumerate(self.users):
                pos = line.find('|')
                uid = int(line[:pos])
                self.uid2id[uid] = idx
                self.id2uid.append(uid)

    def __len__(self):
        return len(self.users)

    def parse(self, line):
        records = line.strip().split("|")
        uid = records[0]
        uid = int(uid)
        features = []
        for i in range(1, len(records)):
            words = records[i].split()
            values = [int(v) for v in words]
            features.append(values)
        return uid,features

    def get(self,uid):
        return self.parse(self.users[self.uid2id[uid]])

    def get_by_id(self,id):
        return self.parse(self.users[id])

class Ads:
    def __init__(self):
        self.ad_fn = None
        self.ads = None
        self.aid2id = {}
        self.id2aid = []

    def read(self, ad_fn):
        self.ad_fn = ad_fn
        with open(self.ad_fn) as f:
            cols = f.readline().strip().split(',')
            self.ads = f.readlines()
            for idx,line in enumerate(self.ads):
                pos = line.find(',')
                aid = int(line[:pos])
                self.aid2id[aid] = idx
                self.id2aid.append(aid)

    def __len__(self):
        return len(self.ads)

    def parse(self, line):
        records = line.strip().split(",")
        aid = records[0]
        aid = int(aid)
        features = []
        for i in range(1, len(records)):
            # use a list to make it consistent with user
            values = [int(records[i])]
            features.append(values)
        return aid,features

    def get(self,aid):
        return self.parse(self.ads[self.aid2id[aid]])

    def get_by_id(self,id):
        return self.parse(self.ads[id])

if __name__=="__main__":
    users = Users()
    feat_fn = "data/userFeature_clean.data"
    attr_fn = "data0/user_attr_list.txt"
    users.read(feat_fn,attr_fn)
    user = users.get(26325489)
    print(user)
    print(len(users))
    print(users.attr_list)
    ads = Ads()
    ad_feat_fn = "data/adFeature.csv"
    ads.read(ad_feat_fn)
    ad = ads.get(ads.id2aid[0])
    print(ad)
    print(len(ads))
    print(ads.attr_list)