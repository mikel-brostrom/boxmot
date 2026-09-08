"""The complete time-variant workflow produces an evaluable cached dataset."""

from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.config import load_dataset_config
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.validation import DatasetValidationError
from boxmot.engine.dataset_variants.workflow import main
from boxmot.engine.eval.evaluator import eval_setup
from boxmot.engine.materialization.catalog import catalog_mot_dataset
from tests.unit.engine.dataset_variants.test_cache import _parent_build, _raw_catalog


def test_generated_variant_aligns_raw_ground_truth_timestamps_and_reused_perception(tmp_path: Path) -> None:
    _, config_path = _raw_catalog(tmp_path, "parent", tuple(range(60)), timed=False)
    sequence = tmp_path / "parent" / "train" / "seq"
    original_gt = [f"{frame},{1 + frame % 3},{frame / 4},1,4,5,1,1,0.75" for frame in range(1, 61)]
    original_det = [f"{frame},-1,{frame / 4},1,4,5,0.99,-1,-1,-1" for frame in range(1, 61)]
    (sequence / "gt" / "gt.txt").write_text("\n".join(original_gt) + "\n", encoding="utf-8")
    (sequence / "det").mkdir()
    (sequence / "det" / "det.txt").write_text("\n".join(original_det) + "\n", encoding="utf-8")
    parent_catalog = catalog_mot_dataset(load_dataset_config(config_path), split="train", data_root=tmp_path)
    parent = _parent_build(tmp_path, parent_catalog, optional=True)
    parent_manifest_hash = sha256_file(parent / "manifest.json")

    result = main(
        Namespace(
            dataset=config_path,
            split="train",
            sequence="seq",
            build=parent,
            build_root=tmp_path / "variants-cache",
            data_root=tmp_path,
            name="workflow-variant",
            seed=0,
        )
    )

    config = load_dataset_config(result.dataset_config)
    catalog = catalog_mot_dataset(config, split="variable", data_root=tmp_path)
    report = json.loads(result.report_path.read_text())
    selection = report["frames"]
    count = len(selection)
    assert 2 < count < 60
    assert result.statistics["source_frames"] == 60
    assert result.statistics["retained_frames"] == count
    assert report["derived_build_id"] == result.build_path.name
    assert report["derived_catalog_digest"] == catalog.fingerprint
    assert report["source"]["build_id"] == parent.name
    assert sha256_file(parent / "manifest.json") == parent_manifest_hash
    assert [sample.frame_index for sample in catalog.samples] == list(range(count))
    assert [sample.timestamp_s for sample in catalog.samples] == [row["timestamp_s"] for row in selection]
    intervals = np.diff([row["timestamp_s"] for row in selection])
    assert np.unique(intervals.round(10)).size > 1
    assert np.all(intervals > 0.0)
    assert selection[0]["source_frame_id"] == 1
    assert selection[-1]["source_frame_id"] == 60

    generated_sequence = result.dataset_config.parent / "variable" / "seq"
    for directory, originals in (("gt", original_gt), ("det", original_det)):
        rows = (generated_sequence / directory / f"{directory}.txt").read_text().splitlines()
        expected = [
            ",".join((str(target_id), *originals[row["source_frame_id"] - 1].split(",")[1:]))
            for target_id, row in enumerate(selection, start=1)
        ]
        assert rows == expected
    with (generated_sequence / "timestamps.csv").open(newline="", encoding="utf-8") as handle:
        timestamps = list(csv.DictReader(handle))
    assert [int(row["frame_id"]) for row in timestamps] == list(range(1, count + 1))
    assert [float(row["timestamp_s"]) for row in timestamps] == [row["timestamp_s"] for row in selection]

    manifest = DatasetManifest.load(result.build_path)
    assert manifest.metadata["timestamps_digest"] == catalog.metadata["timestamps_digest"]
    assert (
        manifest.metadata["component_fingerprints"] == DatasetManifest.load(parent).metadata["component_fingerprints"]
    )
    cached = list(CachedVisionDataset(result.build_path, load_images=True, load_embeddings=True, load_masks=True))
    assert len(cached) == count
    for frame, selected, source_sample in zip(cached, selection, catalog.samples, strict=True):
        original_index = selected["source_frame_id"] - 1
        assert frame.timestamp_s == original_index / 10.0
        assert source_sample.source_sha256 == parent_catalog.samples[original_index].source_sha256
        assert int(frame.frame.image[0, 0, 0]) == original_index
        assert len(frame.detections) == (0 if original_index == 2 else 1)
        if len(frame.detections):
            assert frame.detections.embeddings[0, -1].item() == original_index + 1.0

    evaluation = Namespace(
        dataset=result.dataset_config,
        experiment=None,
        build=result.build_path,
        data_root=tmp_path,
        split="variable",
    )
    eval_setup(evaluation)
    assert evaluation._build_validated is True
    assert evaluation.seq_info == {"seq": count}
    assert evaluation.gt_folder / "seq" / "gt" / "gt.txt" == generated_sequence / "gt" / "gt.txt"


def test_corrupt_parent_cache_is_rejected_before_creating_raw_variant(tmp_path: Path) -> None:
    parent_catalog, config_path = _raw_catalog(tmp_path, "parent", tuple(range(30)), timed=False)
    parent = _parent_build(tmp_path, parent_catalog, optional=True)
    manifest = DatasetManifest.load(parent)
    shard = parent / manifest.artifact("embeddings").shards[0].path
    with shard.open("ab") as handle:
        handle.write(b"corrupted")

    with pytest.raises(DatasetValidationError, match="byte count differs"):
        main(
            Namespace(
                dataset=config_path,
                split="train",
                sequence="seq",
                build=parent,
                build_root=tmp_path / "variants-cache",
                data_root=tmp_path,
                name="corrupt-parent-variant",
                seed=0,
            )
        )

    assert not (tmp_path / "variants").exists()
    assert not (tmp_path / "variants-cache").exists()
