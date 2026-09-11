"""End-to-end KITTI build replay through box and optional segmentation evaluation."""

from __future__ import annotations

import builtins
import importlib
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from boxmot.datasets.config import load_dataset_config
from boxmot.datasets.schema import INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, instance_records, mask_records
from boxmot.engine.eval import catalog_cache, evaluator
from boxmot.engine.eval.mots_io import read_mots_results
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build
from boxmot.engine.materialization.catalog import SourceCatalog, catalog_mot_dataset
from boxmot.structures import Boxes, Detections, MaskBatch


@pytest.fixture(autouse=True)
def _isolated_catalog_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep evaluation's source metadata cache in the test workspace."""
    monkeypatch.setattr(
        catalog_cache, "default_source_metadata_cache_path", lambda _root: tmp_path / "source-cache.json"
    )


def _dataset_fixture(tmp_path: Path, indices: tuple[int, ...]) -> tuple[Path, np.ndarray]:
    """Create two tiny KITTI sequences with cars, pedestrians, and ignored pixels."""
    labels = np.zeros((24, 32), dtype=np.uint16)
    labels[2:10, 2:12] = 1001
    labels[4:7, 5:9] = 0  # A hole prevents bounding boxes from substituting for the segmentation mask.
    labels[12:22, 20:26] = 2001
    labels[0, :] = 10000
    for sequence in ("0000", "0001"):
        images = tmp_path / "KITTI-fixture" / "images" / sequence
        annotations = tmp_path / "KITTI-fixture" / "instances" / sequence
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)
        for index in indices:
            assert cv2.imwrite(str(images / f"{index:06d}.png"), np.zeros((24, 32, 3), dtype=np.uint8))
            assert cv2.imwrite(str(annotations / f"{index:06d}.png"), labels)
    path = tmp_path / "mots-fixture.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "mots-fixture",
                "format": {"layout": "sequence", "box_type": "aabb"},
                "storage": {"root": "KITTI-fixture"},
                "default_split": "train",
                "fps": 10,
                "modalities": {
                    "images": {"format": "image-directory", "path": "images/{sequence}"},
                    "ground_truth": {
                        "format": "instance-png",
                        "path": "instances/{sequence}",
                        "options": {"class_divisor": 1000, "background_id": 0, "ignore_ids": [10000]},
                    },
                },
                "splits": {
                    "train": {"partition": "training", "has_ground_truth": True},
                    "test": {"partition": "testing", "has_ground_truth": False, "modalities": {"ground_truth": None}},
                },
                "classes": {"target": {"car": 1, "pedestrian": 2}, "ignore": {"ignore": 10}},
            }
        ),
        encoding="utf-8",
    )
    return path, labels


def _publish_build(
    tmp_path: Path,
    catalog: SourceCatalog,
    labels: np.ndarray,
    *,
    publish_masks: bool,
    prediction_masks: np.ndarray | None = None,
) -> Path:
    """Publish canonical detections with optional full-frame prediction masks."""
    detect = StagePlan.create("detect", component={"id": "perfect-mots-fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="mots-fixture",
        box_type="aabb",
        source_fingerprint=catalog.fingerprint,
        publish=PublishOptions(image_references=True, masks=publish_masks),
        stages=(detect, finalize),
        metadata=catalog.metadata,
    )
    samples, instances, masks = [], [], []
    for sample in catalog.samples:
        samples.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "sequence_id": sample.sequence_id,
                "frame_index": sample.frame_index,
                "timestamp_s": sample.timestamp_s,
                "image_ref": sample.image_ref,
                "height": sample.image_size[0],
                "width": sample.image_size[1],
            }
        )
        detections = Detections(
            geometry=Boxes(torch.tensor([[2, 2, 12, 10], [20, 12, 26, 22]], dtype=torch.float32)),
            scores=torch.tensor([0.95, 0.95], dtype=torch.float32),
            class_ids=torch.tensor([1, 2], dtype=torch.int64),
            sample_id=sample.sample_id,
            masks=MaskBatch(
                torch.from_numpy(
                    np.stack([labels == 1001, labels == 2001]) if prediction_masks is None else prediction_masks
                )
            ),
        ).with_instance_ids(tuple(f"{plan.build_id}:{sample.sample_id}:{index}" for index in range(2)))
        instances.extend(instance_records(detections, build_id=plan.build_id))
        if publish_masks:
            masks.extend(mask_records(detections))
    writer = ParquetShardWriter(plan.staging_root, box_type="aabb")
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    if publish_masks:
        writer.write(MASKS_ARTIFACT, masks, shard_index=0)
    return finalize_build(plan)


def _args_fixture(
    tmp_path: Path,
    *,
    indices: tuple[int, ...] = (0, 1, 2),
    fps: float | None = None,
    publish_masks: bool = True,
    eval_masks: bool = False,
    prediction_masks: np.ndarray | None = None,
) -> tuple[Namespace, SourceCatalog, np.ndarray]:
    """Resolve a real dataset catalog and matching build for the public evaluator."""
    dataset_path, labels = _dataset_fixture(tmp_path, indices)
    catalog = catalog_mot_dataset(load_dataset_config(dataset_path), split="train", data_root=tmp_path, fps=fps)
    build = _publish_build(tmp_path, catalog, labels, publish_masks=publish_masks, prediction_masks=prediction_masks)
    return (
        Namespace(
            dataset=dataset_path,
            experiment=None,
            data_root=tmp_path,
            split="train",
            build=build,
            fps=fps,
            eval_masks=eval_masks,
            tracker="bytetrack",
            tracker_backend="python",
            sequence_workers=1,
            project=tmp_path / "evaluation",
            name="fixture",
        ),
        catalog,
        labels,
    )


@pytest.mark.parametrize(
    ("native_indices", "fps", "evaluation_indices", "timeline_length"),
    [
        ((0, 1, 2), None, (0, 1, 2), 3),
        ((0, 1, 2), 5.0, (0, 1), 2),
        ((4, 6, 8), None, (4, 6, 8), 9),
    ],
)
def test_kitti_mots_eval_replays_published_masks_and_matches_exact_annotation_frames(
    tmp_path: Path,
    native_indices: tuple[int, ...],
    fps: float | None,
    evaluation_indices: tuple[int, ...],
    timeline_length: int,
) -> None:
    api = pytest.importorskip("pycocotools.mask")
    args, catalog, labels = _args_fixture(tmp_path, indices=native_indices, fps=fps, eval_masks=True)

    result = evaluator.run_eval(args, verbose=False, show_progress=False)

    assert result.summary_label == "cls_comb_det_av"
    assert result.timings["frames"] == len(catalog.samples) == 2 * len(evaluation_indices)
    assert args.sequence_frame_counts == {"0000": len(evaluation_indices), "0001": len(evaluation_indices)}
    assert args.seq_info == {"0000": timeline_length, "0001": timeline_length}
    for class_name in ("car", "pedestrian"):
        metrics = result.raw[class_name]
        for metric in ("HOTA", "MOTA", "MOTP", "sMOTA", "IDF1"):
            assert metrics[metric] == pytest.approx(100)
        assert metrics["GT_Dets"] == metrics["Dets"] == len(catalog.samples)
        assert metrics["Frames"] == 2 * timeline_length
        assert metrics["GT_IDs"] == metrics["IDs"] == 2
        assert set(metrics["per_sequence"]) == {"0000", "0001"}
    assert {path.name for path in result.exp_dir.glob("*.txt")} == {"0000.txt", "0001.txt"}
    for sequence in ("0000", "0001"):
        rows = read_mots_results(
            result.exp_dir / f"{sequence}.txt", frame_shapes={index: labels.shape for index in evaluation_indices}
        )
        assert tuple(rows) == evaluation_indices
        for frame_rows in rows.values():
            assert len(frame_rows) == 2
            for row in frame_rows:
                np.testing.assert_array_equal(api.decode(row.rle), labels == (row.class_id * 1000 + 1))
        gt_frames = args.evaluation_config["mots_gt_frames"][sequence]
        assert tuple(frame[0] for frame in gt_frames) == evaluation_indices
        expected_filenames = (0, 2) if fps is not None else native_indices
        assert tuple(int(Path(frame[1]).stem) for frame in gt_frames) == expected_filenames


@pytest.mark.parametrize("publish_masks", [False, True])
@pytest.mark.parametrize(
    ("native_indices", "fps", "evaluation_indices", "timeline_length"),
    [
        ((0, 1, 2), None, (0, 1, 2), 3),
        ((0, 1, 2), 5.0, (0, 1), 2),
        ((4, 6, 8), None, (4, 6, 8), 9),
    ],
)
def test_kitti_default_box_eval_supports_builds_with_or_without_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish_masks: bool,
    native_indices: tuple[int, ...],
    fps: float | None,
    evaluation_indices: tuple[int, ...],
    timeline_length: int,
) -> None:
    """A box tracker evaluates tight GT boxes without loading the optional COCO codec."""
    args, catalog, _ = _args_fixture(tmp_path, indices=native_indices, fps=fps, publish_masks=publish_masks)
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def import_without_coco(name: str, *args, **kwargs):
        if name == "pycocotools" or name.startswith("pycocotools."):
            pytest.fail("Default KITTI box evaluation must not import pycocotools")
        return original_import(name, *args, **kwargs)

    def import_module_without_coco(name: str, *args, **kwargs):
        if name == "pycocotools" or name.startswith("pycocotools."):
            pytest.fail("Default KITTI box evaluation must not import pycocotools")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_coco)
    monkeypatch.setattr(importlib, "import_module", import_module_without_coco)

    result = evaluator.run_eval(args, verbose=False, show_progress=False)

    assert args.eval_masks is False
    assert result.summary_label == "cls_comb_det_av"
    assert result.timings["frames"] == len(catalog.samples) == 2 * len(evaluation_indices)
    assert args.sequence_frame_counts == {"0000": len(evaluation_indices), "0001": len(evaluation_indices)}
    assert args.seq_info == {"0000": timeline_length, "0001": timeline_length}
    for class_name in ("car", "pedestrian"):
        metrics = result.raw[class_name]
        for metric in ("HOTA", "MOTA", "MOTP", "sMOTA", "IDF1"):
            assert metrics[metric] == pytest.approx(100)
        assert metrics["GT_Dets"] == metrics["Dets"] == len(catalog.samples)
        assert metrics["Frames"] == 2 * timeline_length
        assert metrics["GT_IDs"] == metrics["IDs"] == 2
        assert set(metrics["per_sequence"]) == {"0000", "0001"}
    for sequence in ("0000", "0001"):
        rows = np.loadtxt(result.exp_dir / f"{sequence}.txt", delimiter=",", ndmin=2)
        assert rows.shape == (2 * len(evaluation_indices), 9)
        np.testing.assert_array_equal(np.unique(rows[:, 0]), np.array(evaluation_indices) + 1)
        for frame_number in np.unique(rows[:, 0]):
            frame_rows = rows[rows[:, 0] == frame_number]
            frame_rows = frame_rows[np.argsort(frame_rows[:, 7])]
            np.testing.assert_allclose(frame_rows[:, 2:6], [[2, 2, 10, 8], [20, 12, 6, 10]])
            np.testing.assert_array_equal(frame_rows[:, 7], [1, 2])


def test_kitti_same_build_evaluates_boxes_and_masks_with_different_overlap_scores(tmp_path: Path) -> None:
    """A filled prediction matches the GT box exactly but misses its segmentation hole."""
    pytest.importorskip("pycocotools.mask")
    prediction_masks = np.zeros((2, 24, 32), dtype=np.bool_)
    prediction_masks[0, 2:10, 2:12] = True
    prediction_masks[1, 12:22, 20:26] = True
    args, _, _ = _args_fixture(tmp_path, indices=(0,), prediction_masks=prediction_masks)

    box_result = evaluator.run_eval(args, output_dir=tmp_path / "boxes", verbose=False, show_progress=False)
    args.eval_masks = True
    mask_result = evaluator.run_eval(args, output_dir=tmp_path / "masks", verbose=False, show_progress=False)

    for metric in ("MOTA", "MOTP", "sMOTA"):
        assert box_result.raw["car"][metric] == pytest.approx(100)
        assert box_result.raw["pedestrian"][metric] == pytest.approx(100)
        assert mask_result.raw["pedestrian"][metric] == pytest.approx(100)
    assert mask_result.raw["car"]["MOTA"] == pytest.approx(100)
    assert mask_result.raw["car"]["MOTP"] == pytest.approx(85)
    assert mask_result.raw["car"]["sMOTA"] == pytest.approx(85)
    assert box_result.timings["frames"] == mask_result.timings["frames"] == 2


def test_kitti_mots_build_without_published_masks_is_rejected_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _args_fixture(tmp_path, publish_masks=False, eval_masks=True)

    def unexpected_replay(*_args, **_kwargs):
        pytest.fail("A build without masks must be rejected before tracker replay")

    monkeypatch.setattr(evaluator, "replay_build", unexpected_replay)
    with pytest.raises(ValueError, match="mask"):
        evaluator.run_eval(args, verbose=False)


def test_kitti_mots_unannotated_split_is_rejected_before_build_loading(tmp_path: Path) -> None:
    dataset_path, _ = _dataset_fixture(tmp_path, (0,))
    args = Namespace(dataset=dataset_path, experiment=None, split="test", build="absent", data_root=tmp_path)

    with pytest.raises(ValueError, match="no evaluation ground truth"):
        evaluator.eval_setup(args)


@pytest.mark.parametrize("eval_masks", [False, True])
def test_kitti_mots_kalman_calibration_reports_unsupported_mask_ground_truth(tmp_path: Path, eval_masks: bool) -> None:
    dataset_path, _ = _dataset_fixture(tmp_path, (0,))
    args = Namespace(
        dataset=dataset_path,
        experiment=None,
        split="train",
        build="absent",
        data_root=tmp_path,
        calibrate_kf=True,
        eval_masks=eval_masks,
    )

    with pytest.raises(ValueError, match="Kalman calibration does not support KITTI MOTS"):
        evaluator.eval_setup(args)


def test_kitti_mots_tuning_trials_reuse_build_masks_and_selected_sequence(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    args, _, labels = _args_fixture(tmp_path, eval_masks=True)
    args.sequence_names = ("0001",)
    build_bytes = {path.relative_to(args.build): path.read_bytes() for path in args.build.rglob("*") if path.is_file()}

    first = evaluator.run_eval(args, evolve_config={"track_thresh": 0.4}, verbose=False)
    second = evaluator.run_eval(args, evolve_config={"track_thresh": 0.7}, setup=False, verbose=False)

    assert first.exp_dir != second.exp_dir
    assert first.exp_dir.parent.name == second.exp_dir.parent.name == "trials"
    assert first.raw == second.raw
    assert first.timings["frames"] == second.timings["frames"] == 3
    assert args.sequence_frame_counts == args.seq_info == {"0001": 3}
    for result in (first, second):
        assert {path.name for path in result.exp_dir.glob("*.txt")} == {"0001.txt"}
        assert result.raw["car"]["HOTA"] == result.raw["pedestrian"]["HOTA"] == 100
        rows = read_mots_results(result.exp_dir / "0001.txt", frame_shapes={index: labels.shape for index in range(3)})
        assert len(rows) == 3
        assert set(result.raw["car"]["per_sequence"]) == {"0001"}
    assert build_bytes == {
        path.relative_to(args.build): path.read_bytes() for path in args.build.rglob("*") if path.is_file()
    }
