#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import os
from torch.utils.data import dataloader, distributed

from .datasets import TrainValDataset
from ..utils.events import LOGGER
from ..utils.torch_utils import torch_distributed_zero_first


def create_dataloader(
    path,
    img_size,
    batch_size,
    stride,
    hyp=None,
    augment=False,
    check_images=False,
    check_labels=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    shuffle=False,
    data_dict=None,
    task="Train",
):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    if rect and shuffle:
        LOGGER.warning(
            "WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False"
        )
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = TrainValDataset(
            path,
            img_size,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            check_images=check_images,
            check_labels=check_labels,
            stride=int(stride),
            pad=pad,
            rank=rank,
            data_dict=data_dict,
            task=task,
        )

    batch_size = min(batch_size, len(dataset))
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    return (
        TrainValDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
            collate_fn=TrainValDataset.collate_fn,
        ),
        dataset,
    )


class TrainValDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
