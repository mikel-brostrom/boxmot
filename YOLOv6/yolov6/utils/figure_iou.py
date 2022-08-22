#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch


class IOUloss:
    """ Calculate IoU loss.
    """
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        box2 = box2.T
        if self.box_format == 'xyxy':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif self.box_format == 'xywh':
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def pairwise_bbox_iou(box1, box2, box_format='xywh'):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    if box_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)

    elif box_format == 'xywh':
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)
