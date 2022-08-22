#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import cv2
import numpy as np
import math
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import YOLOv6.tools.eval as eval
from YOLOv6.yolov6.data.data_load import create_dataloader
from YOLOv6.yolov6.models.yolo import build_model
from YOLOv6.yolov6.models.loss import ComputeLoss
from YOLOv6.yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg
from YOLOv6.yolov6.utils.ema import ModelEMA, de_parallel
from YOLOv6.yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from YOLOv6.yolov6.solver.build import build_optimizer, build_lr_scheduler
from YOLOv6.yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from YOLOv6.yolov6.utils.nms import xywh2xyxy


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
        # get model and optimizer
        model = self.get_model(args, cfg, self.num_classes, device)
        if cfg.training_mode == 'repopt':
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            self.optimizer = self.get_optimizer(args, cfg, model)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        self.start_epoch = 0
        # resume
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb

        # set color for classnames
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]

    # Training Process

    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop()

        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_in_loop(self):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                self.train_in_steps()
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    # Training loop for batchdata
    def train_in_steps(self):
        images, targets = self.prepro_data(self.batch_data, self.device)

        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            self.plot_train_batch(images, targets)
            write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')

        # forward
        with amp.autocast(enabled=self.device != 'cpu'):
            preds = self.model(images)
            total_loss, loss_items = self.compute_loss(preds, targets)
            if self.rank != -1:
                total_loss *= self.world_size
        # backward
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 1
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])  # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[0] * 0.1 + self.evaluate_results[1] * 0.9
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            ckpt = {
                'model': deepcopy(de_parallel(self.model)).half(),
                'ema': deepcopy(self.ema.ema).half(),
                'updates': self.ema.updates,
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
            }

            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            del ckpt

            # log for learning rate
            lr = [x['lr'] for x in self.optimizer.param_groups]
            self.evaluate_results = list(self.evaluate_results) + lr

            # log for tensorboard
            write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)

            # save validation predictions to tensorboard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths

        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im

        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column

                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color,
                                    thickness=1)

        self.vis_train_batch = mosaic.copy()

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()  # xyxy
            ori_img = cv2.imread(vis_path)

            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])

                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]),
                              thickness=1)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))

    def eval_model(self):
        results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                   batch_size=self.batch_size // self.world_size * 2,
                                                   img_size=self.img_size,
                                                   model=self.ema.ema,
                                                   dataloader=self.val_loader,
                                                   save_dir=self.save_dir,
                                                   task='train')

        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]

        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)

    def train_before_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        self.best_ap, self.ap = 0.0, 0.0
        self.evaluate_results = (0, 0)  # AP50, AP50_95
        self.compute_loss = ComputeLoss(iou_type=self.cfg.model.head.iou_type)

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(4, device=self.device)
        self.optimizer.zero_grad()

        LOGGER.info(('\n' + '%10s' * 5) % ('Epoch', 'iou_loss', 'l1_loss', 'obj_loss', 'cls_loss'))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + '%10.4g' * 4) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                 *(self.mean_loss)))

    # Empty cache if training finished
    def train_after_loop(self):
        if self.main_process:
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val')[0]

        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    def get_model(self, args, cfg, nc, device):
        model = build_model(cfg, nc, device)
        weights = cfg.model.pretrained
        if cfg.training_mode == 'repvgg' and weights:
            if weights:  # finetune if pretrained model is set
                LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
                model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        return model

    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.model.scales
        scales = None
        if not weights:
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            ckpt = torch.load(weights, map_location=device)
            scales = extract_scales(ckpt)
        return scales

    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model

    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf
