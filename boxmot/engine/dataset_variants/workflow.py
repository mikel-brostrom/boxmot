"""Create a MOT frame-loss variant and reuse its parent perception cache."""

from __future__ import annotations

import configparser
import csv
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from boxmot.datasets import DatasetManifest
from boxmot.datasets.config import load_dataset_config, resolve_dataset_storage_root
from boxmot.datasets.validation import validate_published_build
from boxmot.engine.dataset_variants.cache import derive_cached_build
from boxmot.engine.dataset_variants.sampling import PROFILE, select_bursty_frames, timing_statistics
from boxmot.engine.materialization.builds import resolve_build_path, validate_build_compatibility
from boxmot.engine.materialization.catalog import (
    SourceCatalog,
    catalog_mot_dataset,
    default_data_root,
    resolve_dataset_annotation_root,
    resolve_dataset_split_root,
)
from boxmot.engine.materialization.plan import default_build_root
from boxmot.engine.materialization.source import _local_path
from boxmot.utils import logger as LOGGER
from boxmot.utils.config import validate_config_id


@dataclass(frozen=True, slots=True)
class RawTimeVariant:
    """Published raw data and its exact mapping back to the source catalog."""

    dataset_config: Path
    report_path: Path
    sample_map: Mapping[str, str]
    statistics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TimeVariantResult:
    """Locations required to replay or tune the generated timestamped split."""

    dataset_config: Path
    report_path: Path
    build_path: Path
    statistics: Mapping[str, Any]


def _remap_mot_rows(source: Path, frame_map: Mapping[int, int], frame_count: int) -> tuple[str, int]:
    """Retain complete MOT annotation rows while replacing only frame IDs."""
    selected = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split(",")
        if len(columns) < 7:
            raise ValueError(f"{source}:{line_number} is not a MOT annotation row.")
        try:
            values = [float(value) for value in columns]
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number} contains non-numeric annotations.") from exc
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{source}:{line_number} contains non-finite annotations.")
        source_id = values[0]
        if not source_id.is_integer() or not 1 <= source_id <= frame_count:
            raise ValueError(f"{source}:{line_number} frame ID is outside the source sequence.")
        target_id = frame_map.get(int(source_id))
        if target_id is not None:
            selected.append(",".join((str(target_id), *columns[1:])))
    return "".join(line + "\n" for line in selected), len(selected)


def create_raw_variant(
    dataset: Mapping[str, Any],
    parent_catalog: SourceCatalog,
    *,
    sequence: str,
    name: str | None = None,
    data_root: str | Path | None = None,
    seed: int = 0,
    parent_build_id: str,
) -> RawTimeVariant:
    """Copy selected source images and aligned annotations into a new dataset.

    Existing outputs are refused. A temporary sibling directory keeps partial
    image/annotation writes out of the final dataset location.
    """
    if dataset["layout"] != "mot" or dataset["box_type"] != "aabb":
        raise ValueError("materialize --time-variant currently requires a MOT-layout AABB dataset.")
    originals = tuple(sample for sample in parent_catalog.samples if sample.sequence_id == sequence)
    if not originals:
        available = ", ".join(sorted({sample.sequence_id for sample in parent_catalog.samples}))
        raise ValueError(f"Unknown sequence {sequence!r}; available: {available}.")
    originals = tuple(sorted(originals, key=lambda sample: sample.frame_index))
    if tuple(sample.frame_index for sample in originals) != tuple(range(len(originals))):
        raise ValueError("Source sequence frame indices must be contiguous and start at zero.")
    split = str(parent_catalog.metadata["split"])
    if not dataset["splits"][split]["has_ground_truth"]:
        raise ValueError("materialize --time-variant requires a source split with ground truth.")
    selection = select_bursty_frames([sample.timestamp_s for sample in originals], seed=seed)
    timestamps = [float(sample.timestamp_s) for sample in originals]
    statistics = timing_statistics(timestamps, selection)
    dataset_name = validate_config_id(name or f"{sequence.lower()}-variable-time", path="--name", label="dataset")
    relative_root = f"variants/{dataset_name}"
    output_root = resolve_dataset_storage_root({"root": relative_root}, data_root)
    if output_root.exists():
        raise FileExistsError(f"Variant already exists: {output_root}. Choose a different --name.")
    source_sequence = resolve_dataset_split_root(dataset, split, data_root) / sequence
    source_sequence.resolve().relative_to(parent_catalog.source_root)
    annotations = resolve_dataset_annotation_root(dataset, split, data_root) / sequence
    source_gt = annotations / "gt" / "gt.txt"
    if not source_gt.is_file():
        raise FileNotFoundError(f"Source MOT ground truth is missing: {source_gt}.")
    frame_map = {source_index + 1: target_index + 1 for target_index, source_index in enumerate(selection.indices)}
    gt_text, gt_rows = _remap_mot_rows(source_gt, frame_map, len(originals))
    source_det = source_sequence / "det" / "det.txt"
    det_text = _remap_mot_rows(source_det, frame_map, len(originals))[0] if source_det.is_file() else None
    selected = [originals[index] for index in selection.indices]
    if len({sample.image_size for sample in selected}) != 1:
        raise ValueError("A MOT sequence must have constant image dimensions.")
    suffixes = {_local_path(sample.source_uri).suffix.lower() for sample in selected}
    if len(suffixes) != 1 or any(sample.source_frame_index is not None for sample in selected):
        raise ValueError("materialize --time-variant requires still source images with a common extension.")
    extension = next(iter(suffixes))
    sample_map = {f"variable:{sequence}:{index}": sample.sample_id for index, sample in enumerate(selected)}
    classes = {
        group: {label: value["id"] for label, value in dataset["classes"].items() if value["evaluation"] == group}
        for group in ("target", "ignore")
    }
    config = {
        "id": dataset_name,
        "format": {"layout": "mot", "box_type": "aabb"},
        "storage": {"root": relative_root},
        "default_split": "variable",
        "splits": {"variable": {"path": "variable", "has_ground_truth": True}},
        "classes": classes,
    }
    report = {
        "profile": PROFILE,
        "seed": seed,
        "description": "Real source footage with simulated throttling and frame loss; capture times are unchanged.",
        "time_unit": "seconds",
        "timestamp_source": "source catalog (timestamps.csv if present; otherwise seqinfo.ini frame rate)",
        "source": {
            "dataset_id": dataset["id"],
            "split": split,
            "sequence": sequence,
            "catalog_digest": parent_catalog.fingerprint,
            "build_id": parent_build_id,
        },
        "statistics": {**statistics, "ground_truth_rows": gt_rows},
        "outage": {
            "source_frame_start": selection.outage_start_index + 1,
            "source_frame_end_exclusive": selection.outage_end_index + 1,
            "start_s": timestamps[selection.outage_start_index],
            "end_s": timestamps[selection.outage_end_index],
        },
        "frames": [
            {
                "frame_id": index + 1,
                "source_frame_id": sample.frame_index + 1,
                "source_sample_id": sample.sample_id,
                "timestamp_s": sample.timestamp_s,
            }
            for index, sample in enumerate(selected)
        ],
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset_name}-", dir=output_root.parent) as temporary:
        staging = Path(temporary) / "dataset"
        target_sequence = staging / "variable" / sequence
        image_root = target_sequence / "img1"
        image_root.mkdir(parents=True)
        for index, sample in enumerate(selected, start=1):
            shutil.copy2(_local_path(sample.source_uri), image_root / f"{index:06d}{extension}")
        (target_sequence / "gt").mkdir()
        (target_sequence / "gt" / "gt.txt").write_text(gt_text, encoding="utf-8")
        if det_text is not None:
            (target_sequence / "det").mkdir()
            (target_sequence / "det" / "det.txt").write_text(det_text, encoding="utf-8")
        with (target_sequence / "timestamps.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame_id", "timestamp_s"))
            writer.writerows((index, sample.timestamp_s) for index, sample in enumerate(selected, start=1))
        info = configparser.ConfigParser()
        info.optionxform = str
        info.read(source_sequence / "seqinfo.ini")
        if not info.has_section("Sequence"):
            info.add_section("Sequence")
        height, width = selected[0].image_size
        info["Sequence"].update(
            {
                "name": sequence,
                "imDir": "img1",
                "seqLength": str(len(selected)),
                "imWidth": str(width),
                "imHeight": str(height),
                "imExt": extension,
            }
        )
        with (target_sequence / "seqinfo.ini").open("w", encoding="utf-8") as handle:
            info.write(handle)
        (staging / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (staging / "variant.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if output_root.exists():
            raise FileExistsError(f"Variant already exists: {output_root}. Choose a different --name.")
        staging.rename(output_root)
    return RawTimeVariant(output_root / "dataset.yaml", output_root / "variant.json", sample_map, statistics)


def main(args: Any) -> TimeVariantResult:
    """Create raw data, publish reused perception, and display replay locations."""
    dataset = load_dataset_config(args.dataset)
    data_root = default_data_root(args.data_root)
    parent_catalog = catalog_mot_dataset(dataset, split=args.split, data_root=data_root)
    parent_build = resolve_build_path(args.build, build_root=args.build_root)
    manifest = DatasetManifest.load(parent_build)
    validate_build_compatibility(
        manifest,
        dataset_id=str(dataset["id"]),
        split=args.split,
        geometry=str(dataset["box_type"]),
        source_catalog_digest=parent_catalog.fingerprint,
        class_taxonomy_digest=str(parent_catalog.metadata["class_taxonomy_digest"]),
    )
    validate_published_build(parent_build, manifest=manifest)
    raw = create_raw_variant(
        dataset,
        parent_catalog,
        sequence=args.sequence,
        name=args.name,
        data_root=data_root,
        seed=args.seed,
        parent_build_id=manifest.build_id,
    )
    catalog = catalog_mot_dataset(load_dataset_config(raw.dataset_config), split="variable", data_root=data_root)
    build_path = derive_cached_build(
        parent_build,
        parent_catalog=parent_catalog,
        catalog=catalog,
        sample_map=raw.sample_map,
        build_root=default_build_root() if args.build_root is None else Path(args.build_root),
    )
    report = json.loads(raw.report_path.read_text(encoding="utf-8"))
    report["derived_build_id"] = build_path.name
    report["derived_catalog_digest"] = catalog.fingerprint
    raw.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stats = raw.statistics
    LOGGER.info(
        f"Created {PROFILE}: {stats['retained_frames']}/{stats['source_frames']} frames, "
        f"{stats['duration_s']:.3f}s, dt {1000 * stats['min_dt_s']:.1f}–{1000 * stats['max_dt_s']:.1f}ms."
    )
    LOGGER.info(f"Dataset: {raw.dataset_config}\nBuild: {build_path}\nReport: {raw.report_path}")
    return TimeVariantResult(raw.dataset_config, raw.report_path, build_path, stats)
