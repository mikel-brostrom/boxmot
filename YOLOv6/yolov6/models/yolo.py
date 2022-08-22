#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch.nn as nn
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import EfficientRep
from yolov6.models.reppan import RepPANNeck
from yolov6.models.effidehead import Detect, build_effidehead_layer


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.mode = config.training_mode
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.detect(x)
        return x

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_anchors = config.model.head.anchors
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)

    backbone = EfficientRep(
        in_channels=channels,
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    neck = RepPANNeck(
        channels_list=channels_list,
        num_repeats=num_repeat,
        block=block
    )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers)

    return backbone, neck, head


def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    return model
