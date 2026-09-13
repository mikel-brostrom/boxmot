"""Shared split selection, annotation pairing, and reports for KITTI MOTS replay."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import yaml

from boxmot.configs import CONFIG_ROOT
from boxmot.engine.eval.mots import run_mots_metrics
from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.datasets.sensor_cache import SensorReplaySequence

GroundTruthEntry = tuple[int, str, int, int]


def kitti_mots_sequences(split: str, selected: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve the repository's MOTS split, rejecting accidental split mixing."""
    config = yaml.safe_load((CONFIG_ROOT / "datasets" / "kitti-mots.yaml").read_text(encoding="utf-8"))
    if split not in {"train", "val", "fulltrain"}:
        raise ValueError("KITTI MOTS local evaluation requires split train, val, or fulltrain with ground truth.")
    available = tuple(config["splits"][split].get("sequences", [f"{index:04d}" for index in range(21)]))
    if len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError(f"Sequences must be distinct members of KITTI MOTS {split}: {', '.join(available)}")
    return tuple(name for name in available if not selected or name in selected)


def kitti_mots_annotations(
    sequence_id: str,
    frame_paths: Sequence[Path],
    image_size: tuple[int, int],
    ground_truth: Path,
) -> list[GroundTruthEntry]:
    """Pair image frames with absolute annotation paths from their sequence directory."""
    entries = []
    for image_path in frame_paths:
        annotation = ground_truth / image_path.with_suffix(".png").name
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing KITTI MOTS ground-truth instance PNG: {annotation}")
        entries.append((int(image_path.stem), str(annotation.resolve()), *image_size))
    return entries


def evaluate_kitti_mots(
    prediction_dir: Path,
    output: Path,
    instances_root: Path,
    annotations: Mapping[str, Sequence[GroundTruthEntry]],
    *,
    cached_ground_truth: Mapping[str, SensorReplaySequence] | None = None,
    ground_truth_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate disjoint predicted masks and write combined/per-sequence reports."""
    LOGGER.info("Evaluating segmentation tracking against KITTI MOTS instance annotations")
    options = {} if cached_ground_truth is None else {"cached_ground_truth": cached_ground_truth}
    if ground_truth_options is not None:
        options["ground_truth_options"] = ground_truth_options
    results = run_mots_metrics(
        SimpleNamespace(exp_dir=prediction_dir, evaluation_config={"mots_gt_frames": annotations}),
        [Path(name) for name in annotations],
        output,
        instances_root,
        seq_info={name: max(entry[0] for entry in entries) + 1 for name, entries in annotations.items()},
        **options,
    )
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fields = [key for key in results["car"] if key != "per_sequence"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", *fields])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"class": name, **{key: values[key] for key in fields}})
            LOGGER.info(
                f"{name}: HOTA {values['HOTA']:.2f} | DetA {values['DetA']:.2f} | "
                f"AssA {values['AssA']:.2f} | IDF1 {values['IDF1']:.2f}"
            )
    return results


__all__ = ("evaluate_kitti_mots", "kitti_mots_annotations", "kitti_mots_sequences")
