# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch.nn as nn
from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords,
                                  lr=config.TRAIN.BASE_LR)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), lr=None):
    assert lr
    has_decay = []
    no_decay = []
    skip_keywords_prefix = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            skip_keywords_prefix.append(name)
            continue
        if hasattr(module, 'no_weight_decay'):
            nwd_dict = module.no_weight_decay()
            for post_fix in nwd_dict:
                if name == '':
                    whole_name = post_fix
                else:
                    whole_name = name + '.' + post_fix
                skip_keywords_prefix.append(whole_name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords) or check_keywords_match_name_prefix(name, skip_keywords_prefix):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_keywords_match_name_prefix(name: str, keywords=()):
    isin = False
    for keyword in keywords:
        if name.startswith(keyword):
            isin = True
    return isin
