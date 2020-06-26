import math
import sys
import time
import torch
import torch.nn as nn
from .yolo_utils import bbox_iou, multi_bbox_ious, convert2cpu


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.rescore = 0
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.
        self.stride = 32
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0

    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        masked_anchors = torch.FloatTensor(masked_anchors).to(self.device)
        num_anchors = torch.IntTensor([len(self.anchor_mask)]).to(self.device)
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        nB = target.size(0)
        anchor_step = anchors.size(1)  # anchors[nA][anchor_step]
        conf_mask = torch.ones(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
        twidth, theight = self.net_width / self.stride, self.net_height / self.stride

        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0

        # it works faster on CPU than on GPU.
        anchors = anchors.to("cpu")

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5).to("cpu")
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = cur_ious > self.ignore_thresh
            conf_mask[b][ignore_ix.view(nA, nH, nW)] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors), 1).t()
                _, best_n = torch.max(multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False), 0)

                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = 1
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.

                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1

        return nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        mask_tuple = self.get_mask_boxes(output)
        t0 = time.time()
        nB = output.data.size(0)  # batch size
        nA = mask_tuple['n'].item()  # num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0) // nA
        anchors = mask_tuple['a'].view(nA, anchor_step).to(self.device)
        cls_anchor_dim = nB * nA * nH * nW

        output = output.view(nB, nA, (5 + nC), nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to(self.device)
        ix = torch.LongTensor(range(0, 5)).to(self.device)
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to(self.device)

        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1,
                                                                                                             cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()  # x, y
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)

        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim).to(
            self.device)
        anchor_w = anchors.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)

        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()

        t2 = time.time()
        nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)

        cls_mask = (cls_mask == 1)
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).to(self.device)
        cls = cls[cls_mask].view(-1, nC)

        nProposals = int((conf > 0.25).sum())

        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf, tcls = tconf.to(self.device), tcls.to(self.device)
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim).to(self.device), conf_mask.to(self.device)

        t3 = time.time()
        loss_coord = nn.MSELoss(size_average=False)(coord * coord_mask, tcoord * coord_mask) / 2
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask)
        loss_cls = nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        loss = loss_coord + loss_conf + loss_cls

        t4 = time.time()
        if False:
            print('-' * 30)
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print(
            '%d: Layer(%03d) nGT %3d, nRC %3d, nRC75 %3d, nPP %3d, loss: box %6.3f, conf %6.3f, class %6.3f, total %7.3f'
            % (self.seen, self.nth_layer, nGT, nRecall, nRecall75, nProposals, loss_coord, loss_conf, loss_cls, loss))
        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss
