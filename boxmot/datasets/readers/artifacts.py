"""Decode canonical Parquet artifacts into BoxMOT structures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from boxmot.structures import Boxes, Detections, MaskBatch, OrientedBoxes

from ..masks import MASK_CODEC, _validate_mask_payload, unpack_mask_batch
from ..schema import (
    INSTANCES_ARTIFACT,
    INSTANCES_PATH,
    MASKS_ARTIFACT,
    MASKS_PATH,
    SAMPLES_ARTIFACT,
    SAMPLES_PATH,
    BoxType,
)
from ..storage import read_parquet_artifact


def read_sample_records(root: str | Path) -> dict[str, dict[str, Any]]:
    """Read canonical sample rows keyed by ``sample_id``."""

    rows = read_parquet_artifact(Path(root) / SAMPLES_PATH, artifact_name=SAMPLES_ARTIFACT).to_pylist()
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Sample artifacts contain duplicate sample IDs.")
    return by_id


def read_detection_batches(root: str | Path, box_type: BoxType) -> dict[str, Detections]:
    """Read canonical instance rows as detection batches keyed by sample ID."""

    samples = read_sample_records(root)
    instance_rows = read_parquet_artifact(
        Path(root) / INSTANCES_PATH,
        artifact_name=INSTANCES_ARTIFACT,
        box_type=box_type,
    ).to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        grouped[row["sample_id"]].append(row)

    detections: dict[str, Detections] = {}
    for sample_id in samples:
        rows = sorted(grouped.get(sample_id, ()), key=lambda row: row["detection_index"])
        if box_type == "aabb":
            values = [[row[key] for key in ("x1", "y1", "x2", "y2")] for row in rows]
            geometry = Boxes(torch.tensor(values, dtype=torch.float32).reshape(-1, 4).contiguous())
        else:
            values = [[row[key] for key in ("cx", "cy", "w", "h", "angle")] for row in rows]
            geometry = OrientedBoxes(torch.tensor(values, dtype=torch.float32).reshape(-1, 5).contiguous())
        detections[sample_id] = Detections(
            geometry=geometry,
            scores=torch.tensor([row["score"] for row in rows], dtype=torch.float32).contiguous(),
            class_ids=torch.tensor([row["class_id"] for row in rows], dtype=torch.int64).contiguous(),
            sample_id=sample_id,
            instance_ids=tuple(row["instance_id"] for row in rows),
        )
    return detections


def attach_masks(root: str | Path, detections: dict[str, Detections]) -> dict[str, Detections]:
    """Decode and key-join a canonical mask artifact onto detection batches."""

    samples = read_sample_records(root)
    rows = read_parquet_artifact(Path(root) / MASKS_PATH, artifact_name=MASKS_ARTIFACT).to_pylist()
    row_keys = [(row["sample_id"], row["instance_id"]) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("Mask artifacts contain duplicate sample/instance keys.")
    expected_keys = {
        (sample_id, instance_id)
        for sample_id, batch in detections.items()
        for instance_id in (batch.instance_ids or ())
    }
    if set(row_keys) != expected_keys:
        raise ValueError("Mask artifact keys must match detection keys exactly once.")
    keyed = dict(zip(row_keys, rows, strict=True))
    for (sample_id, instance_id), row in keyed.items():
        sample = samples[sample_id]
        if (row["height"], row["width"]) != (sample["height"], sample["width"]):
            raise ValueError(f"Mask {instance_id!r} is not full-frame sized.")
        if row["codec"] != MASK_CODEC:
            raise ValueError(f"Mask {instance_id!r} uses an unsupported codec.")
        _validate_mask_payload(row["data"], row["height"], row["width"])
    enriched: dict[str, Detections] = {}
    for sample_id, batch in detections.items():
        sample = samples[sample_id]
        payloads = [keyed[(sample_id, instance_id)]["data"] for instance_id in batch.instance_ids or ()]
        values = unpack_mask_batch(payloads, sample["height"], sample["width"])
        enriched[sample_id] = batch.with_masks(MaskBatch(values))
    return enriched


__all__ = ("attach_masks", "read_detection_batches", "read_sample_records")
