import numpy as np
import logging

def adjust_learning_rate(lr, optimizer, cur_batch, lr_steps, lr_decay=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = lr_decay ** (sum(cur_batch > np.array(lr_steps)))
    lr = lr * decay
    if cur_batch in np.array(lr_steps)+1:
        logging.info("cur_batch [{}] set learning rate to {}".format(cur_batch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_mountain(base_lr, optimizer, cur_batch, end, mul=10):
    if cur_batch<=0.5*end:
        lr = base_lr*(mul-1)*(cur_batch/(end*0.5))+base_lr
    else:
        lr = base_lr*(mul-1)*((end-cur_batch)/(end*0.5))+base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

def adjust_learning_rate_custom(base_lr, optimizer, cur_batch, func):
    '''

    :param lr:
    :param optimizer:
    :param cur_batch:
    :param func:
    :return:
    '''
    lr = func(base_lr,cur_batch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']