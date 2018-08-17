# import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method("spawn")

import warnings
warnings.simplefilter('once',UserWarning)

# load configuration
from lib.tools import parse_args,load_config
args = parse_args()
config_fn = args.cfg
cfg = load_config(config_fn,args)

# random seed
import torch
import numpy as np
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
assert cfg["cuda"],"only support gpu"
if cfg["cuda"]:
    torch.cuda.manual_seed(cfg["seed"])

import time
timestamp = time.strftime('%Y-%m-%d-%H-%M')

# create logger
import logging
from lib.logger import create_logger
_ = create_logger(cfg["output_path"],cfg["comment"],timestamp)

# backup config
import shutil
import os
shutil.copyfile(args.cfg, os.path.join(cfg["output_path"],"{}_{}".format(timestamp, cfg["config_fn"])))

# set env
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x),cfg["gpu_ids"]))

from optimizer.adam import Adam
from eval import cal_avg_auc
from dataset.dataset_youth_r import DatasetYouth
from dataset.samplers import RadioSampler
from lib.tools import load_users_and_ads,load_data_list,load_feature_infos,load_rfeats
from lib.checkpoint import load_checkpoint,save_checkpoint
from lib.lr_scheduler import adjust_learning_rate
from meter import AverageMeter,RunningValue,SumMeter,AUCMeter
from torch.utils.data import DataLoader
#from dataset.dataloader import DataLoader
#from dataset.dataiter import get_data_iter
from torch.nn.utils.clip_grad import clip_grad_norm
from model import *

import time

# train
import pprint

logging.info("Configuration:")
logging.info(pprint.pformat(cfg))


def main():
    # load users, ads and their information of features
    users, ads, u_feat_infos, a_feat_infos = load_users_and_ads(cfg["data"]["user_fn"],
                                                                cfg["data"]["ad_fn"],
                                                                cfg["data"]["user_fi_fn"],
                                                                cfg["data"]["ad_fi_fn"],
                                                                )


    r_feat_infos = load_feature_infos(cfg["data"]["r_fi_fp"])

    logging.info("There are {} users.".format(len(users)))
    logging.info("There are {} ads.".format(len(ads)))

    # load data list and history features
    if not args.test:
        train_list = load_data_list(cfg["train_fp"])
        #print("train list len:",len(train_list))
        train_rfeats = load_rfeats(cfg["data"]["train_rfeat_fp"])
        valid_list = load_data_list(cfg["valid_fp"])
        valid_rfeats = load_rfeats(cfg["data"]["valid_rfeat_fp"])
    else:
        test_list = load_data_list(cfg["test_fp"])
        test_rfeats = load_rfeats(cfg["data"]["test_rfeat_fp"])

    filter = cfg["feat"]["filter"]

    # construct mappng and filter
    [fi.construct_mapping() for fi in u_feat_infos]
    [fi.construct_mapping() for fi in a_feat_infos]
    [fi.construct_mapping() for fi in r_feat_infos]

    # filter out low-frequency features.
    for fi in u_feat_infos:
        if fi.name in filter:
            fi.construct_filter(l_freq=filter[fi.name])
        else:
            fi.construct_filter(l_freq=0)
    logging.warning("Users Filtering!!!")

    for fi in a_feat_infos:
        if fi.name in filter:
            fi.construct_filter(l_freq=filter[fi.name])
        else:
            fi.construct_filter(l_freq=0)
    logging.warning("Ads Filtering!!!")

    reg = cfg["reg"]

    if not args.test:
        train_dataset = DatasetYouth(users,u_feat_infos,
                                    ads,a_feat_infos,
                                    train_rfeats, r_feat_infos,
                                    train_list,
                                    cfg["feat"]["u_enc"],
                                    cfg["feat"]["a_enc"],
                                    cfg["feat"]["r_enc"],
                                    reg = reg,
                                    pos_weight=cfg["train"]["pos_weight"],
                                    has_label=True)

        #print("train num: ",train_dataset.original_len)

        if cfg["train"]["use_radio_sampler"]:
            radio_sampler = RadioSampler(train_dataset,p2n_radio=cfg["train"]["p2n_radio"])
            logging.info("Using radio sampler with p:n={}".format(cfg["train"]["p2n_radio"]))

        valid_dataset = DatasetYouth(users,u_feat_infos,
                                    ads,a_feat_infos,
                                    valid_rfeats, r_feat_infos,
                                    valid_list,
                                    cfg["feat"]["u_enc"],
                                    cfg["feat"]["a_enc"],
                                    cfg["feat"]["r_enc"],
                                    reg = reg,
                                    pos_weight=cfg["train"]["pos_weight"],
                                    has_label=True)

        dataset = train_dataset

    else:
        test_dataset = DatasetYouth(users,u_feat_infos,
                                    ads,a_feat_infos,
                                    test_rfeats, r_feat_infos,
                                    test_list,
                                    cfg["feat"]["u_enc"],
                                    cfg["feat"]["a_enc"],
                                    cfg["feat"]["r_enc"],
                                    reg = reg,
                                    has_label=False)

        dataset = test_dataset

    logging.info("shuffle: {}".format(False if cfg["train"]["use_radio_sampler"] else True))



    # set up model
    emedding_cfgs = {}
    emedding_cfgs.update(cfg["feat"]["u_embed_cfg"])
    emedding_cfgs.update(cfg["feat"]["a_embed_cfg"])

    loss_cfg = cfg["loss"]

    # create model
    model = eval(cfg["model_name"])(n_out=1,
                                    u_embedding_feat_infos=dataset.embedding_u_feat_infos,
                                    u_one_hot_feat_infos=dataset.one_hot_u_feat_infos,
                                    a_embedding_feat_infos=dataset.embedding_a_feat_infos,
                                    a_one_hot_feat_infos=dataset.one_hot_a_feat_infos,
                                    r_embedding_feat_infos=dataset.embedding_r_feat_infos,
                                    embedding_cfgs=emedding_cfgs,
                                    loss_cfg=loss_cfg,
                                    )


    # model = DataParallel(model,device_ids=cfg["gpus"])
    # logging.info("Using model {}.".format(cfg["model_name"]))

    ## optmizers
    # todo lr,weight decay
    optimizer = Adam(model.get_train_policy(), lr=cfg["optim"]["lr"],weight_decay=cfg["optim"]["weight_decay"], amsgrad=True)
    #optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum=0.9,weight_decay=cfg["optim"]["weight_decay"])
    logging.info("Using optimizer {}.".format(optimizer))


    if cfg["train"]["resume"] or args.test:
        checkpoint_file = cfg["resume_fp"]
        state = load_checkpoint(checkpoint_file)
        logging.info("Load checkpoint file {}.".format(checkpoint_file))
        st_epoch = state["cur_epoch"]+1
        logging.info("Start from {}th epoch.".format(st_epoch))
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
    else:
        st_epoch = 1
    ed_epoch = cfg["train"]["ed_epoch"]

    # move tensor to gpu and wrap tensor with Variable
    to_gpu_variable = dataset.get_to_gpu_variable_func()


    if args.extract_weight:
        model = model.module
        path = os.path.join(cfg["output_path"],"weight")
        os.makedirs(path,exist_ok=True)
        u_embedder = model.u_embedder
        u_embedder.save_weight(path)
        a_embedder = model.a_embedder
        a_embedder.save_weight(path)
        exit(0)

    def evaluate(output, label, label_weights):
        '''
        Note the input to this function should be converted to data first.
        :param output:
        :param label_weights:
        :param target:
        :return:
        '''
        output = output.view(-1)
        label = label.view(-1).byte()
        #print(output[0:100])
        #print(label[0:100])
        scores = torch.sigmoid(output)
        output = scores>0.1
        #print(output)
        # print(label.float().sum())
        tp = ((output==label)*label).float().sum()
        fp = ((output!=label)*output).float().sum()
        fn = ((output!=label)*(1-output)).float().sum()
        tn = ((output==label)*(1-label)).float().sum()
        return tp,fp,fn,tn,scores.cpu()

    def valid(cur_train_epoch, phase="valid", extract_features=False):
        '''

        :param cur_train_epoch:
        :param phase: "valid" or "test"
        :return:
        '''
        assert phase in ["valid","test"]

        results = []
        valid_detail_meters = {
            "loss": SumMeter(),
            "model_loss": SumMeter(),
            "tp": SumMeter(),
            "fn": SumMeter(),
            "fp": SumMeter(),
            "tn": SumMeter(),
            "batch_time": AverageMeter(),
            "io_time": AverageMeter(),
        }

        if phase=="valid":
            logging.info("Valid data.")
            dataset = valid_dataset
        else:
            logging.info("Test data.")
            dataset = test_dataset

        model.eval()
        logging.info("Set network to eval model")


        if extract_features:
            features =  np.zeros(shape=(dataset.original_len,model.n_output_feat),dtype=np.float32)
            features_ctr = 0

        batch_idx = 0

        # chunked here
        chunk_size = 200
        n_chunk = (dataset.original_len+(cfg[phase]["batch_size"]*chunk_size)-1)//(cfg[phase]["batch_size"]*chunk_size)
        n_batch = (dataset.original_len+cfg[phase]["batch_size"]-1)//cfg[phase]["batch_size"]


        for chunk_idx in range(n_chunk):
            s = chunk_idx*cfg[phase]["batch_size"]*chunk_size
            e = (chunk_idx+1)*cfg[phase]["batch_size"]*chunk_size

            dataloader = DataLoader(dataset.slice(s,e),
                                      batch_size=cfg[phase]["batch_size"],
                                      shuffle=False,
                                      num_workers=cfg[phase]["n_worker"],
                                      collate_fn=dataset.get_collate_func(),
                                      pin_memory=True,
                                      drop_last=False,
                                      )

            batch_time_s = time.time()
            for samples in dataloader:
                batch_idx = batch_idx + 1
                cur_batch = batch_idx
                valid_detail_meters["io_time"].update(time.time()-batch_time_s)

                # move to gpu
                samples = to_gpu_variable(samples,volatile=True)

                # forward
                loss, output, model_loss, reg_loss, d = model(samples)

                if phase == "valid":

                    # evaluate metrics
                    valid_detail_meters["loss"].update(loss.data[0]*samples["size"],samples["size"])
                    valid_detail_meters["model_loss"].update(model_loss.data[0]*samples["size"],samples["size"])
                    tp, fp, fn, tn, scores = evaluate(output.data, samples["labels"].data, samples["label_weights"].data)
                    #print(tp,fn,fp,tn)
                    valid_detail_meters["tp"].update(tp,samples["size"])
                    valid_detail_meters["fp"].update(fp,samples["size"])
                    valid_detail_meters["fn"].update(fn,samples["size"])
                    valid_detail_meters["tn"].update(tn,samples["size"])
                    # the large the better
                    tp_rate = valid_detail_meters["tp"].sum/(valid_detail_meters["tp"].sum+valid_detail_meters["fn"].sum+1e-20)
                    # the smaller the better
                    fp_rate = valid_detail_meters["fp"].sum/(valid_detail_meters["fp"].sum+valid_detail_meters["tn"].sum+1e-20)
                    valid_detail_meters["batch_time"].update(time.time() - batch_time_s)
                    batch_time_s = time.time()
                else:
                    scores = torch.sigmoid(output.data)
                    valid_detail_meters["batch_time"].update(time.time() - batch_time_s)
                    batch_time_s = time.time()

                # collect results
                uids = samples["uids"]
                aids = samples["aids"]
                results.extend(zip(aids,uids,scores))


                # collect features
                if extract_features:
                    bs = samples["size"]
                    features[features_ctr:features_ctr+bs,:] = d.data.cpu().numpy()
                    features_ctr += bs

                # log results
                if phase=="valid":
                    if cur_batch % cfg["valid"]["logging_freq"] == 0:
                        logging.info("Valid Batch [{cur_batch}/{ed_batch}] "
                                     "loss: {loss} "
                                     "model_loss: {model_loss} "
                                     "tp: {tp} fn: {fn} fp: {fp} tn: {tn} "
                                     "tp_rate: {tp_rate} fp_rate: {fp_rate} "
                                     "io time: {io_time}s batch time {batch_time}s".format(
                            cur_batch=cur_batch,
                            ed_batch=n_batch,
                            loss=valid_detail_meters["loss"].mean,
                            model_loss=valid_detail_meters["model_loss"].mean,
                            tp = valid_detail_meters["tp"].sum,
                            fn=valid_detail_meters["fn"].sum,
                            fp = valid_detail_meters["fp"].sum,
                            tn = valid_detail_meters["tn"].sum,
                            tp_rate = tp_rate,
                            fp_rate = fp_rate,
                            io_time=valid_detail_meters["io_time"].mean,
                            batch_time=valid_detail_meters["batch_time"].mean,
                        )
                        )
                else:
                    if cur_batch % cfg["test"]["logging_freq"] == 0:
                        logging.info("Test Batch [{cur_batch}/{ed_batch}] "
                                     "io time: {io_time}s batch time {batch_time}s".format(
                            cur_batch=cur_batch,
                            ed_batch=n_batch,
                            io_time=valid_detail_meters["io_time"].mean,
                            batch_time=valid_detail_meters["batch_time"].mean,
                        )
                        )

        if phase=="valid":
            logging.info("{phase} for {cur_train_epoch} train epoch "
                         "loss: {loss} "
                         "model_loss: {model_loss} "
                         "tp_rate: {tp_rate} fp_rate: {fp_rate} "
                         "io time: {io_time}s batch time {batch_time}s".format(
                        phase=phase,
                        cur_train_epoch=cur_train_epoch,
                        loss=valid_detail_meters["loss"].mean,
                        model_loss=valid_detail_meters["model_loss"].mean,
                        tp_rate=tp_rate,
                        fp_rate=fp_rate,
                        io_time=valid_detail_meters["io_time"].mean,
                        batch_time=valid_detail_meters["batch_time"].mean,
                    )
                    )

            # write results to file
            res_fn = "{}_{}".format(cfg["valid_res_fp"], cur_train_epoch)
            with open(res_fn, 'w') as f:
                f.write("aid,uid,score\n")
                for res in results:
                    f.write("{},{},{:.8f}\n".format(res[0], res[1], res[2]))
            # evaluate results
            avg_auc, aucs = cal_avg_auc(res_fn, cfg["valid_fp"])
            logging.info("Valid for {cur_train_epoch} train epoch "
                         "average auc {avg_auc}".format(
                cur_train_epoch=cur_train_epoch,
                avg_auc=avg_auc,
            )
            )
            logging.info("aucs: ")
            logging.info(pprint.pformat(aucs))

        else:
            logging.info("Test for {} train epoch ends.".format(cur_train_epoch))

            res_fn = "{}_{}".format(cfg["test_res_fp"], cur_train_epoch)
            with open(res_fn, 'w') as f:
                f.write("aid,uid,score\n")
                for res in results:
                    f.write("{},{},{:.8f}\n".format(res[0], res[1], res[2]))

        # extract features
        if extract_features:
            import pickle as pkl
            with open(cfg["extracted_features_fp"],"wb") as f:
                pkl.dump(features,f,protocol=pkl.HIGHEST_PROTOCOL)

    model.cuda()
    logging.info("Move network to gpu.")

    if args.test:
        valid(st_epoch-1,phase="test",extract_features=args.extract_features)
        exit(0)
    elif cfg["valid"]["init_valid"]:
        valid(st_epoch-1)
        model.train()
        logging.info("Set network to train model.")

    # train: main loop

    model.train()
    logging.info("Set network to train model.")

    # original_lambda = cfg["reg"]["lambda"]
    total_n_batch = 0
    warnings.warn("total_n_batch always start at 0...")

    for cur_epoch in range(st_epoch,ed_epoch+1):

        # meters
        k = cfg["train"]["logging_freq"]
        detail_meters = {
            "loss": RunningValue(k),
            "epoch_loss": SumMeter(),
            "model_loss": RunningValue(k),
            "epoch_model_loss": SumMeter(),
            "tp": RunningValue(k),
            "fn": RunningValue(k),
            "fp": RunningValue(k),
            "tn": RunningValue(k),
            "auc": AUCMeter(),
            "batch_time": AverageMeter(),
            "io_time": AverageMeter(),
        }

        # adjust lr
        adjust_learning_rate(cfg["optim"]["lr"], optimizer, cur_epoch, cfg["train"]["lr_steps"], lr_decay=cfg["train"]["lr_decay"])

        # dynamic adjust cfg["reg"]["lambda"]
        # decay = 1/(0.5 ** (sum(cur_epoch > np.array(cfg["train"]["lr_steps"]))))
        # cfg["reg"]["lambda"] = original_lambda * decay
        # print("using dynamic regularizer, {}".format(cfg["reg"]["lambda"]))


        train_dataset.shuffle()

        batch_idx = -1

        # chunked here because of memory issue. we always create new DataLoader after several batches.
        chunk_size = 200
        n_chunk = (train_dataset.original_len+(cfg["train"]["batch_size"]*chunk_size)-1)//(cfg["train"]["batch_size"]*chunk_size)
        n_batch = (train_dataset.original_len+cfg["train"]["batch_size"]-1)//cfg["train"]["batch_size"]

        for chunk_idx in range(n_chunk):
            s = chunk_idx*cfg["train"]["batch_size"]*chunk_size
            e = (chunk_idx+1)*cfg["train"]["batch_size"]*chunk_size

            train_dataloader = DataLoader(train_dataset.slice(s,e),
                                          batch_size=cfg["train"]["batch_size"],
                                          shuffle=False if cfg["train"]["use_radio_sampler"] else True,
                                          num_workers=cfg["train"]["n_worker"],
                                          collate_fn=train_dataset.get_collate_func(),
                                          sampler=radio_sampler if cfg["train"]["use_radio_sampler"] else None,
                                          pin_memory=True,
                                          drop_last=True,
                                          )

            batch_time_s = time.time()
            for samples in train_dataloader:
                total_n_batch += 1
                batch_idx = batch_idx+1
                detail_meters["io_time"].update(time.time()-batch_time_s)

                # move to gpu
                samples = to_gpu_variable(samples)

                # forward
                loss, output, model_loss, reg_loss, d = model(samples)

                #print("reg_loss",reg_loss)

                # clear grads
                optimizer.zero_grad()

                # backward
                loss.backward()

                # This is a little useful
                warnings.warn("Using gradients clipping")
                clip_grad_norm(model.parameters(),max_norm=5)

                # update weights
                optimizer.step()

                # evaluate metrics
                detail_meters["loss"].update(loss.data[0])
                detail_meters["epoch_loss"].update(loss.data[0])
                detail_meters["model_loss"].update(model_loss.data[0])
                detail_meters["epoch_model_loss"].update(model_loss.data[0])
                tp,fp,fn,tn,scores = evaluate(output.data,samples["labels"].data,samples["label_weights"].data)
                #print(tp,fn,fp,tn)
                detail_meters["tp"].update(tp)
                detail_meters["fp"].update(fp)
                detail_meters["fn"].update(fn)
                detail_meters["tn"].update(tn)
                # the large the better
                tp_rate = detail_meters["tp"].sum / (detail_meters["tp"].sum + detail_meters["fn"].sum + 1e-20)
                # the smaller the better
                fp_rate = detail_meters["fp"].sum / (detail_meters["fp"].sum + detail_meters["tn"].sum + 1e-20)
                detail_meters["batch_time"].update(time.time()-batch_time_s)

                # collect results
                uids = samples["uids"]
                aids = samples["aids"]
                preds = zip(aids, uids, scores)
                gts = zip(aids,uids,samples["labels"].cpu().data)
                detail_meters["auc"].update(preds,gts)

                batch_time_s = time.time()

                # log results
                if (batch_idx+1) % cfg["train"]["logging_freq"]==0:
                    logging.info("Train Batch [{cur_batch}/{ed_batch}] "
                                 "loss: {loss} "
                                 "model_loss: {model_loss} "
                                 "auc: {auc} "
                                 "tp: {tp} fn: {fn} fp: {fp} tn: {tn} "
                                 "tp_rate: {tp_rate} fp_rate: {fp_rate} "
                                 "io time: {io_time}s batch time {batch_time}s".format(
                        cur_batch=batch_idx+1,
                        ed_batch=n_batch,
                        loss = detail_meters["loss"].mean,
                        model_loss = detail_meters["model_loss"].mean,
                        tp=detail_meters["tp"].sum,
                        fn=detail_meters["fn"].sum,
                        fp=detail_meters["fp"].sum,
                        tn=detail_meters["tn"].sum,
                        auc = detail_meters["auc"].auc,
                        tp_rate=tp_rate,
                        fp_rate=fp_rate,
                        io_time = detail_meters["io_time"].mean,
                        batch_time = detail_meters["batch_time"].mean,
                    )
                    )
                    detail_meters["auc"].reset()

                if total_n_batch % cfg["train"]["backup_freq_batch"] == 0 and total_n_batch >= cfg["train"]["start_backup_batch"]:
                    state_to_save = {
                        "cur_epoch": cur_epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    checkpoint_file = os.path.join(cfg["output_path"], "epoch_{}_tbatch_{}.checkpoint".format(cur_epoch,total_n_batch))
                    save_checkpoint(state_to_save, checkpoint_file)
                    logging.info("Save checkpoint to {}.".format(checkpoint_file))

                if total_n_batch%cfg["train"]["valid_freq_batch"] == 0 and total_n_batch>=cfg["train"]["start_valid_batch"]:
                    valid(cur_epoch)
                    model.train()
                    logging.info("Set network to train model.")


        logging.info("Train Epoch [{cur_epoch}] "
                     "loss: {loss} "
                     "model_loss: {model_loss} ".format(
            cur_epoch=cur_epoch,
            loss=detail_meters["epoch_loss"].mean,
            model_loss=detail_meters["epoch_model_loss"].mean,
        )
        )

        # back up
        if cur_epoch % cfg["train"]["backup_freq_epoch"]==0 and cur_epoch>=cfg["train"]["start_backup_epoch"]:
            state_to_save = {
                "cur_epoch": cur_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            checkpoint_file = os.path.join(cfg["output_path"],"epoch_{}.checkpoint".format(cur_epoch))
            save_checkpoint(state_to_save,checkpoint_file)
            logging.info("Save checkpoint to {}.".format(checkpoint_file))

        # valid on valid dataset
        if cur_epoch % cfg["train"]["valid_freq_epoch"]==0 and cur_epoch>=cfg["train"]["start_valid_epoch"]:
            valid(cur_epoch)
            model.train()
            logging.info("Set network to train model.")

if __name__=="__main__":
    main()
