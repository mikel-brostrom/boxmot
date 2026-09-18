"""Configurable instance-PNG annotations decoded into canonical tracks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from boxmot.structures import Frame, Tracks


def read_instance_png(
    path: Path,
    frame: Frame,
    *,
    class_ids: Sequence[int],
    class_divisor: int,
    background_id: int,
    ignore_ids: Sequence[int] = (),
    ignore_class_ids: Sequence[int] = (),
) -> tuple[Tracks, torch.Tensor]:
    """Decode class*divisor+instance IDs and preserve configured ignored pixels.

    Full encoded labels become track IDs. Boxes use exclusive maximum pixel
    coordinates, and masks retain their full image dimensions on empty frames.
    """
    import cv2
    import numpy as np
    import torch

    from boxmot.structures import Boxes, MaskBatch, Tracks

    if type(class_divisor) is not int or class_divisor <= 0:
        raise ValueError("instance-png class_divisor must be a positive integer.")
    if type(background_id) is not int or not 0 <= background_id <= 65535:
        raise ValueError("instance-png background_id must be a uint16 integer.")
    if any(type(value) is not int or not 0 <= value <= 65535 for value in ignore_ids):
        raise ValueError("instance-png ignore_ids must contain uint16 integers.")
    if not class_ids or any(type(value) is not int or value < 0 for value in class_ids):
        raise ValueError("instance-png class_ids must contain nonnegative integers.")
    if any(type(value) is not int or value < 0 for value in ignore_class_ids):
        raise ValueError("instance-png ignore_class_ids must contain nonnegative integers.")
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"Unable to decode instance PNG: {path}")
    if labels.dtype != np.uint16 or labels.ndim != 2:
        raise ValueError(f"Instance PNG must be single-channel uint16, got {labels.shape}, {labels.dtype}: {path}")
    if labels.shape != frame.image_size:
        raise ValueError(
            f"Instance PNG dimensions {labels.shape} do not match image dimensions {frame.image_size}: {path}"
        )
    object_ids = np.unique(labels)
    excluded = (
        (object_ids == background_id)
        | np.isin(object_ids, ignore_ids)
        | np.isin(object_ids // class_divisor, ignore_class_ids)
    )
    valid = excluded | np.isin(object_ids // class_divisor, class_ids)
    if not valid.all():
        raise ValueError(f"Instance PNG contains unsupported labels {object_ids[~valid].tolist()}: {path}")
    object_ids = object_ids[~excluded].astype(np.int64)
    masks = labels[None, :, :] == object_ids[:, None, None]
    boxes = np.empty((len(object_ids), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        boxes[index] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    tracks = Tracks(
        geometry=Boxes(torch.from_numpy(boxes)),
        track_ids=torch.from_numpy(object_ids),
        scores=torch.ones(len(object_ids), dtype=torch.float32),
        class_ids=torch.from_numpy(object_ids // class_divisor),
        detection_indices=torch.full((len(object_ids),), -1, dtype=torch.int64),
        sample_id=frame.sample_id,
        masks=MaskBatch(torch.from_numpy(masks)),
    )
    ignored = np.isin(labels, ignore_ids) | np.isin(labels // class_divisor, ignore_class_ids)
    return tracks, torch.from_numpy(ignored)
