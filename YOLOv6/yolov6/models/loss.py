#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# The code is based on
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from YOLOv6.yolov6.utils.figure_iou import IOUloss, pairwise_bbox_iou


class ComputeLoss:
    '''Loss computation func.
    This func contains SimOTA and siou loss.
    '''
    def __init__(self,
                 reg_weight=5.0,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 center_radius=2.5,
                 eps=1e-7,
                 in_channels=[256, 512, 1024],
                 strides=[8, 16, 32],
                 n_anchors=1,
                 iou_type='ciou'
                 ):

        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

        self.center_radius = center_radius
        self.eps = eps
        self.n_anchors = n_anchors
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # Define criteria
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(iou_type=iou_type, reduction="none")

    def __call__(
        self,
        outputs,
        targets
    ):
        dtype = outputs[0].type()
        device = targets.device
        loss_cls, loss_obj, loss_iou, loss_l1 = torch.zeros(1, device=device), torch.zeros(1, device=device), \
            torch.zeros(1, device=device), torch.zeros(1, device=device)
        num_classes = outputs[0].shape[-1] - 5

        outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides = self.get_outputs_and_grids(
            outputs, self.strides, dtype, device)

        total_num_anchors = outputs.shape[1]
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        bbox_preds_org = outputs_origin[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # targets
        batch_size = bbox_preds.shape[0]
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))

        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        num_targets_list = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects

        num_fg, num_gts = 0, 0
        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []

        for batch_idx in range(batch_size):
            num_gt = int(num_targets_list[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:

                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5].mul_(gt_bboxes_scale)
                gt_classes = targets[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        cls_preds_per_image,
                        obj_preds_per_image,
                        expanded_strides,
                        xy_shifts,
                        num_classes
                    )

                except RuntimeError:
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    print("------------CPU Mode for This Batch-------------")

                    _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                    _gt_classes = gt_classes.cpu().float()
                    _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                    _cls_preds_per_image = cls_preds_per_image.cpu().float()
                    _obj_preds_per_image = obj_preds_per_image.cpu().float()

                    _expanded_strides = expanded_strides.cpu().float()
                    _xy_shifts = xy_shifts.cpu()

                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        _gt_bboxes_per_image,
                        _gt_classes,
                        _bboxes_preds_per_image,
                        _cls_preds_per_image,
                        _obj_preds_per_image,
                        _expanded_strides,
                        _xy_shifts,
                        num_classes
                    )

                    gt_matched_classes = gt_matched_classes.cuda()
                    fg_mask = fg_mask.cuda()
                    pred_ious_this_matching = pred_ious_this_matching.cuda()
                    matched_gt_inds = matched_gt_inds.cuda()

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                if num_fg_img > 0:
                    cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), num_classes
                    ) * pred_ious_this_matching.unsqueeze(-1)
                    obj_target = fg_mask.unsqueeze(-1)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]

                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        xy_shifts=xy_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            l1_targets.append(l1_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        # loss
        loss_iou += (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks].T, reg_targets)).sum() / num_fg
        loss_l1 += (self.l1_loss(bbox_preds_org.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg

        loss_obj += (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*1.0)).sum() / num_fg
        loss_cls += (self.bcewithlog_loss(cls_preds.view(-1, num_classes)[fg_masks], cls_targets)).sum() / num_fg

        total_losses = self.reg_weight * loss_iou + loss_l1 + loss_obj + loss_cls
        return total_losses, torch.cat((self.reg_weight * loss_iou, loss_l1, loss_obj, loss_cls)).detach()

    def decode_output(self, output, k, stride, dtype, device):
        grid = self.grids[k].to(device)
        batch_size = output.shape[0]
        hsize, wsize = output.shape[2:4]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype).to(device)
            self.grids[k] = grid

        output = output.reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        output_origin = output.clone()
        grid = grid.view(1, -1, 2)

        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, output_origin, grid, hsize, wsize

    def get_outputs_and_grids(self, outputs, strides, dtype, device):
        xy_shifts = []
        expanded_strides = []
        outputs_new = []
        outputs_origin = []

        for k, output in enumerate(outputs):
            output, output_origin, grid, feat_h, feat_w = self.decode_output(
                output, k, strides[k], dtype, device)

            xy_shift = grid
            expanded_stride = torch.full((1, grid.shape[1], 1), strides[k], dtype=grid.dtype, device=grid.device)

            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            outputs_new.append(output)
            outputs_origin.append(output_origin)

        xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
        expanded_strides = torch.cat(expanded_strides, 1) # [1, n_anchors_all, 1]
        outputs_origin = torch.cat(outputs_origin, 1)
        outputs = torch.cat(outputs_new, 1)

        feat_h *= strides[-1]
        feat_w *= strides[-1]
        gt_bboxes_scale = torch.Tensor([[feat_w, feat_h, feat_w, feat_h]]).type_as(outputs)

        return outputs, outputs_origin, gt_bboxes_scale, xy_shifts, expanded_strides

    def get_l1_target(self, l1_target, gt, stride, xy_shifts, eps=1e-8):

        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        xy_shifts,
        num_classes
    ):

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            xy_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # cost
        pair_wise_ious = pairwise_bbox_iou(gt_bboxes_per_image, bboxes_preds_per_image, box_format='xywh')
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
                * obj_preds_.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_, obj_preds_

        cost = (
            self.cls_weight * pair_wise_cls_loss
            + self.iou_weight * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        xy_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        xy_shifts_per_image = xy_shifts[0] * expanded_strides_per_image
        xy_centers_per_image = (
            (xy_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1, 1)
        )  # [n_anchor, 2] -> [n_gt, n_anchor, 2]

        gt_bboxes_per_image_lt = (
            (gt_bboxes_per_image[:, 0:2] - 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )
        gt_bboxes_per_image_rb = (
            (gt_bboxes_per_image[:, 0:2] + 0.5 * gt_bboxes_per_image[:, 2:4])
            .unsqueeze(1)
            .repeat(1, total_num_anchors, 1)
        )  # [n_gt, 2] -> [n_gt, n_anchor, 2]

        b_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        b_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # in fixed center
        gt_bboxes_per_image_lt = (gt_bboxes_per_image[:, 0:2]).unsqueeze(1).repeat(
            1, total_num_anchors, 1
        ) - self.center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_rb = (gt_bboxes_per_image[:, 0:2]).unsqueeze(1).repeat(
            1, total_num_anchors, 1
        ) + self.center_radius * expanded_strides_per_image.unsqueeze(0)

        c_lt = xy_centers_per_image - gt_bboxes_per_image_lt
        c_rb = gt_bboxes_per_image_rb - xy_centers_per_image
        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
