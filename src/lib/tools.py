from functools import wraps

def timethis(func,*args,**kwargs):
    @wraps(func)
    def wrapper(*args,**kwargs):
        s = time.time()
        ret = func(*args,**kwargs)
        elapse = time.time()-s
        print("{} use {}s".format(func.__name__,elapse))
        return ret
    return wrapper


from data_tool.managers import Users,Ads
from data_tool.feature import FeatureInfo
import yaml
import argparse
import os
import time
import logging

def load_feature_infos(fn,ver=1):
    if ver==1:
        constructor = FeatureInfo
    logging.info("using ver {} featureinfo".format(ver))
    feature_infos = []
    with open(fn) as f:
        for line in f:
            feature_info = constructor()
            feature_info.from_str(line)
            feature_infos.append(feature_info)
    return feature_infos

# def load_feature_infos_v2(fn):
#     feature_infos = []
#     with open(fn) as f:
#         for line in f:
#             feature_info = FeatureInfoV2()
#             feature_info.from_str(line)
#             feature_infos.append(feature_info)
#     return feature_infos

@timethis
def load_rfeats(fn):
    import pandas as pd
    import numpy as np
    # rfeats = pd.read_csv(fn,dtype=np.int32)
    # rfeats = rfeats.values
    # print(rfeats.shape)
    # print(rfeats.dtype)
    # import numpy as np
    # rfeats = np.loadtxt(fn,dtype=np.int32,delimiter=',',skiprows=1)
    rfeats = []
    with open(fn) as f:
        f.readline()
        for line in f:
            records = line.strip().split(',')
            rfeat = [[int(r) for r in record.split()] for record in records]
            rfeats.append(rfeat)
    return rfeats

@timethis
def load_users_and_ads(user_fn,ad_fn,user_feat_fn,ad_feat_fn,ver=1):
    u_feat_infos = load_feature_infos(user_feat_fn,ver)
    a_feat_infos = load_feature_infos(ad_feat_fn,ver)
    users = Users()
    users.read(user_fn)
    ads = Ads()
    ads.read(ad_fn)
    return users,ads,u_feat_infos,a_feat_infos

def load_data_list(data_fn,pred_biases_fn=None):
    if pred_biases_fn is None:
        data_list = []
        with open(data_fn) as f:
            f.readline()
            for line in f:
                records = line.strip().split(',')
                records = [int(r) for r in records]
                data_list.append(tuple(records))
    else:
        data_list = []
        with open(data_fn) as f1, open(pred_biases_fn) as f2:
            f1.readline()
            f2.readline()
            for line in f1:
                line2 = f2.readline()
                records = line.strip().split(',')
                records = [int(r) for r in records]
                pred_bias = float(line2.strip().split(',')[-1])
                records.append(pred_bias)
                data_list.append(tuple(records))
    return data_list


def update_config(config,args):
    config["config_fn"] = os.path.split(args.cfg)[1]
    config["comment"] = args.comment
    return config

def auto_gen_config(config,args):
    # todo automatically generate more config option based on current config
    config["output_path"] = os.path.join(config["root_output_path"],config["data"]["dataset"],config["model_name"],config["description"])
    os.makedirs(config["output_path"], exist_ok=True)

    dataset_path = os.path.join(config["data"]["root_path"],config["data"]["dataset"])
    config["train_fp"] = os.path.join(dataset_path,config["train"]["fn"])
    config["valid_fp"] = os.path.join(dataset_path,config["valid"]["fn"])
    config["test_fp"] = os.path.join(dataset_path,config["test"]["fn"])

    config["data"]["user_fn"] = os.path.join(dataset_path,config["data"]["user_fn"])
    config["data"]["ad_fn"] = os.path.join(dataset_path, config["data"]["ad_fn"])
    config["data"]["user_fi_fn"] = os.path.join(dataset_path, config["data"]["user_fi_fn"])
    config["data"]["ad_fi_fn"] = os.path.join(dataset_path, config["data"]["ad_fi_fn"])
    if "cross_fi_fn" in config["data"]:
        config["data"]["cross_fi_fn"] = os.path.join(dataset_path, config["data"]["cross_fi_fn"])

    config["n_gpus"] = len(config["gpu_ids"])
    config["gpus"] = list(range(config["n_gpus"]))

    config["resume_fp"] = config["resume_fn"]
    config["valid_res_fp"] = os.path.join(config["output_path"],config["valid"]["res_fn"])
    config["test_res_fp"] = os.path.join(config["output_path"], config["test"]["res_fn"])
    if "extracted_features_fn" in config["test"]:
        config["extracted_features_fp"] = os.path.join(config["output_path"],config["test"]["extracted_features_fn"])
    else:
        config["extracted_features_fp"] = os.path.join(config["output_path"], "extracted_features.pkl")

    if "mini_batch" not in config:
        config["mini_batch"] = 1

    # config["train_gbdt_fp"] = os.path.join(dataset_path, config["data"]["gbdt_path"], config["train"]["gbdt_fn"])
    # config["valid_gbdt_fp"] = os.path.join(dataset_path, config["data"]["gbdt_path"], config["valid"]["gbdt_fn"])
    # config["test_gbdt_fp"] = os.path.join(dataset_path, config["data"]["gbdt_path"], config["test"]["gbdt_fn"])

    for embed_cfg in config["feat"]["u_embed_cfg"].values():
        if "atten" not in embed_cfg:
            embed_cfg["atten"] = False

    for embed_cfg in config["feat"]["a_embed_cfg"].values():
        if "atten" not in embed_cfg:
            embed_cfg["atten"] = False

    if "c_embed_cfg" in config["feat"]:
        for embed_cfg in config["feat"]["c_embed_cfg"].values():
            if "atten" not in embed_cfg:
                embed_cfg["atten"] = False

    config["train"]["p2n_radio"] = eval(config["train"]["p2n_radio"])
    return config

def read_config(config_fn):
    '''

    :param config_fn: a yaml file
    :return:
    '''
    with open(config_fn) as f:
        config = yaml.load(f)
    print(config)
    return config

def load_config(config_fn,args):
    config = read_config(config_fn)
    update_config(config,args)
    config = auto_gen_config(config,args)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str,required=True)
    parser.add_argument("--comment",type=str,default="")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--extract_weight",action="store_true")
    parser.add_argument("--extract_features", action="store_true")
    args = parser.parse_args()
    return args




if __name__=="__main__":
    ad_fn = "data/A_shiyu/adFeature.csv"
    ads = Ads()
    ads.read(ad_fn)
    print(len(ads))