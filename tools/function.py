import os
from collections import OrderedDict

import numpy as np
import torch

from tools.utils import may_mkdirs


def seperate_weight_decay(named_params, lr, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        # if 'bias' in name:
        #     no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'lr': lr, 'weight_decay': 0.},
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)

    # --------------------- dangwei li TIP20 ---------------------
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)
    weights[targets > 1] = 0.0

    return weights


def get_model_log_path(root_path, model_name):
    multi_attr_model_dir = os.path.join(root_path, model_name, 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, model_name, 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir



def get_pkl_rootpath(dataset, zero_shot):
    root = os.path.join("./data", f"{dataset}")
    data_path = os.path.join(root, 'dataset_all.pkl')  #

    return data_path


def get_reload_weight(model_path, model, pth='ckpt_max.pth'):
    model_path = os.path.join(model_path, pth)
    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        try:
            pretrain_dict = load_dict['state_dicts']
            print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")
        except:
            pretrain_dict = load_dict['model']

    new_dict = OrderedDict()
    for k,v in pretrain_dict.items():
        k = k.replace('module.', '')
        if k.startswith('head.'):
            k = k.replace('head.', 'classifier.')
        k = k.replace('conv_head.', 'backbone.conv_head.')
        if 'backbone' not in k and 'classifier' not in k:
            k = 'backbone.' + k
        new_dict[k] = v

    pretrain_dict = new_dict

    model.load_state_dict(pretrain_dict, strict=True)

    return model

