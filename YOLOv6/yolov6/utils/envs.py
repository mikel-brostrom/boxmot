#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from YOLOv6.yolov6.utils.events import LOGGER


def get_envs():
    """Get PyTorch needed environments from system envirionments."""
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size


def select_device(device):
    """Set devices' information to the program.
    Args:
        device: a string, like 'cpu' or '1,2,3,4'
    Returns:
        torch.device
    """
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        LOGGER.info('Using CPU for training... ')
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()
        nd = len(device.strip().split(','))
        LOGGER.info(f'Using {nd} GPU for training... ')
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device


def set_random_seed(seed, deterministic=False):
    """ Set random state to random libray, numpy, torch and cudnn.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
