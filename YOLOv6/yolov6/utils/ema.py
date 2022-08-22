#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py
import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay(self.updates)

            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, item in self.ema.state_dict().items():
                if item.dtype.is_floating_point:
                    item *= decay
                    item += (1 - decay) * state_dict[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)


def is_parallel(model):
    # Return True if model's type is DP or DDP, else False.
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model. Return single-GPU model if model's type is DP or DDP.
    return model.module if is_parallel(model) else model
