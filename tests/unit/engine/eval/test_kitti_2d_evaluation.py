"""Image-only KITTI datasets replay real detection builds through eval and tune."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from boxmot.datasets.config import load_dataset_config
from boxmot.datasets.inputs import resolve_sensor_dataset_config_path
from boxmot.datasets.readers import boxes3d, masks
from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, instance_records
from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.eval import catalog_cache, eagermot_kitti, evaluator
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build
from boxmot.engine.materialization.catalog import SourceCatalog, catalog_mot_dataset
from boxmot.engine.tuning import tuner
from boxmot.structures import Boxes, Detections

_BOXES = [[20, 30, 100, 100], [160, 40, 200, 160]]


@pytest.fixture(autouse=True)
def _isolated_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use built-in metrics and forbid unrelated sensor/mask readers."""
    monkeypatch.setitem(sys.modules, "trackeval", None)
    monkeypatch.setattr(
        catalog_cache, "default_source_metadata_cache_path", lambda _root: tmp_path / "source-cache.json"
    )

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("KITTI image-box evaluation must not consume sensor, 3D, or mask inputs.")

    monkeypatch.setattr(eagermot_kitti, "prepare_eagermot_kitti", unexpected)
    monkeypatch.setattr(boxes3d, "read_kitti_tracking_labels", unexpected)
    monkeypatch.setattr(boxes3d, "read_kitti_object_labels", unexpected)
    monkeypatch.setattr(masks, "read_instance_png", unexpected)


def _dataset(tmp_path: Path, frames: tuple[int, ...], *, fps: float | None) -> Path:
    """Use native KITTI paths and unavailable spatial geometry in otherwise valid GT."""
    for sequence in ("0000", "0001"):
        images = tmp_path / "KITTI-2D" / "training" / "image_02" / sequence
        images.mkdir(parents=True)
        annotations = tmp_path / "KITTI-2D" / "training" / "label_02" / f"{sequence}.txt"
        annotations.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in frames:
            assert cv2.imwrite(str(images / f"{frame:06d}.png"), np.zeros((240, 480, 3), dtype=np.uint8))
            # FPS-selected even frames remain static; excluded frames have
            # distinct boxes so an incorrect GT join cannot score perfectly.
            offset = 25 if fps is not None and frame % 2 else 0
            for identity, label, box in zip((7, 8), ("Car", "Pedestrian"), _BOXES, strict=True):
                left, top, right, bottom = box
                rows.append(
                    f"{frame} {identity} {label} 0 0 -10 {left + offset} {top} {right + offset} {bottom} "
                    "-1 -1 -1 -1000 -1000 -1000 -10"
                )
        annotations.write_text("\n".join(rows) + "\n", encoding="utf-8")
    path = tmp_path / "kitti-2d.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "kitti-2d-fixture",
                "format": {"layout": "sequence", "box_type": "aabb"},
                "storage": {"root": "KITTI-2D"},
                "default_split": "train",
                "fps": 10,
                "modalities": {
                    "images": {"format": "image-directory", "path": "{partition}/image_02/{sequence}"},
                    "ground_truth": {"format": "kitti-tracking-labels", "path": "{partition}/label_02/{sequence}.txt"},
                },
                "splits": {"train": {"partition": "training", "has_ground_truth": True}},
                "classes": {"target": {"car": 1, "pedestrian": 2}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _publish_build(tmp_path: Path, catalog: SourceCatalog) -> Path:
    """Publish canonical AABB detections without masks, embeddings, or inference."""
    detect = StagePlan.create("detect", component={"id": "perfect-kitti-2d-fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="kitti-2d-fixture",
        box_type="aabb",
        source_fingerprint=catalog.fingerprint,
        publish=PublishOptions(image_references=True),
        stages=(detect, finalize),
        metadata=catalog.metadata,
    )
    samples, instances = [], []
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
            geometry=Boxes(torch.tensor(_BOXES, dtype=torch.float32)),
            scores=torch.full((2,), 0.95, dtype=torch.float32),
            class_ids=torch.tensor([1, 2], dtype=torch.int64),
            sample_id=sample.sample_id,
        ).with_instance_ids(tuple(f"{plan.build_id}:{sample.sample_id}:{index}" for index in range(2)))
        instances.extend(instance_records(detections, build_id=plan.build_id))
    writer = ParquetShardWriter(plan.staging_root, box_type="aabb")
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    return finalize_build(plan)


def _arguments(
    tmp_path: Path,
    *,
    frames: tuple[int, ...] = (0, 1, 2),
    fps: float | None = None,
    cache_inputs: bool = False,
) -> tuple[Namespace, SourceCatalog]:
    """Resolve a matching dataset/build pair for the production evaluation setup."""
    dataset = _dataset(tmp_path, frames, fps=fps)
    catalog = catalog_mot_dataset(load_dataset_config(dataset), split="train", data_root=tmp_path, fps=fps)
    build = _publish_build(tmp_path, catalog)
    return Namespace(
        dataset=dataset,
        experiment=None,
        data_root=tmp_path,
        split="train",
        build=build,
        fps=fps,
        cache_inputs=cache_inputs,
        tracker="bytetrack",
        tracker_backend="python",
        sequence_workers=1,
        project=tmp_path / "evaluation",
        name="fixture",
    ), catalog


@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize(
    ("frames", "fps", "evaluation_frames", "source_frames", "timeline_length"),
    [
        ((0, 1, 2), None, (0, 1, 2), (0, 1, 2), 3),
        ((0, 1, 2, 3, 4), 5.0, (0, 1, 2), (0, 2, 4), 3),
        ((4, 6, 8), None, (4, 6, 8), (4, 6, 8), 9),
    ],
)
def test_kitti_2d_replays_build_with_exact_annotation_frame_mapping(
    tmp_path: Path,
    cache_inputs: bool,
    frames: tuple[int, ...],
    fps: float | None,
    evaluation_frames: tuple[int, ...],
    source_frames: tuple[int, ...],
    timeline_length: int,
) -> None:
    args, catalog = _arguments(tmp_path, frames=frames, fps=fps, cache_inputs=cache_inputs)
    assert resolve_sensor_dataset_config_path(args.dataset, split="train") is None

    result = evaluator.run_eval(args, verbose=False, show_progress=False)

    assert args.evaluation_config["annotation_layout"] == "kitti_tracking"
    assert args.seq_info == {"0000": timeline_length, "0001": timeline_length}
    assert result.timings["frames"] == len(catalog.samples) == 2 * len(evaluation_frames)
    for sequence in ("0000", "0001"):
        mapping = args.evaluation_config["kitti_gt_sequences"][sequence]
        assert Path(mapping["path"]) == tmp_path / "KITTI-2D/training/label_02" / f"{sequence}.txt"
        assert mapping["frame_count"] == max(frames) + 1
        assert mapping["frames"] == list(zip(evaluation_frames, source_frames, strict=True))
        predictions = np.loadtxt(result.exp_dir / f"{sequence}.txt", delimiter=",", ndmin=2)
        assert predictions.shape == (2 * len(evaluation_frames), 9)
        np.testing.assert_array_equal(np.unique(predictions[:, 0]), np.array(evaluation_frames) + 1)
        assert set(predictions[:, 7]) == {1, 2}
    for class_name in ("car", "pedestrian"):
        metrics = result.raw[class_name]
        for metric in ("HOTA", "MOTA", "IDF1"):
            assert metrics[metric] == pytest.approx(100)
        assert metrics["GT_Dets"] == metrics["Dets"] == len(catalog.samples)
        assert metrics["Frames"] == 2 * timeline_length
        assert metrics["GT_IDs"] == metrics["IDs"] == 2


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_kitti_2d_tuning_trials_use_shared_setup_and_fresh_trackers(tmp_path: Path, cache_inputs: bool) -> None:
    args, _ = _arguments(tmp_path, cache_inputs=cache_inputs)
    args.sequence_names = ("0001",)
    before = {path.relative_to(args.build): path.read_bytes() for path in args.build.rglob("*") if path.is_file()}
    tuner.eval_setup(args)

    first = tuner.run_eval(args, evolve_config={"track_thresh": 0.4}, setup=False, verbose=False, show_progress=False)
    second = tuner.run_eval(args, evolve_config={"track_thresh": 0.99}, setup=False, verbose=False, show_progress=False)

    assert first.exp_dir != second.exp_dir
    assert first.exp_dir.parent.name == second.exp_dir.parent.name == "trials"
    assert args.sequence_frame_counts == args.seq_info == {"0001": 3}
    assert set(args.evaluation_config["kitti_gt_sequences"]) == {"0001"}
    for result in (first, second):
        assert result.timings["frames"] == 3
        assert {path.name for path in result.exp_dir.glob("*.txt")} == {"0001.txt"}
        assert set(result.raw["car"]["per_sequence"]) == {"0001"}
    for class_name in ("car", "pedestrian"):
        assert first.raw[class_name]["HOTA"] == first.raw[class_name]["IDF1"] == 100
        assert second.raw[class_name]["HOTA"] == second.raw[class_name]["Dets"] == 0
        assert first.raw[class_name]["GT_Dets"] == second.raw[class_name]["GT_Dets"] == 3
    assert before == {
        path.relative_to(args.build): path.read_bytes() for path in args.build.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("mode", ["eval", "tune"])
def test_kitti_2d_cli_routes_build_to_image_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    fixture, _ = _arguments(tmp_path)
    captured = {}

    def run_workflow(module: str, args: Namespace) -> Any:
        captured.update(module=module, args=args)
        if mode == "tune":
            # Replace the search scheduler only; run the production shared
            # setup and a real ByteTrack trial using the CLI-built namespace.
            tuner.eval_setup(args)
            result = tuner.run_eval(args, evolve_config={"track_thresh": 0.4}, setup=False, verbose=False)
        else:
            result = evaluator.run_eval(args, verbose=False)
        captured["result"] = result
        return result

    monkeypatch.setattr(_support, "_run_engine_workflow", run_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            mode,
            "--dataset",
            str(fixture.dataset),
            "--build",
            str(fixture.build),
            "--data-root",
            str(tmp_path),
            "--split",
            "train",
            "--tracker",
            "bytetrack",
            "--sequence",
            "0001",
            "--sequence-workers",
            "1",
            "--project",
            str(tmp_path / "cli-evaluation"),
            "--cache-inputs",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["module"] == ("boxmot.engine.eval.evaluator" if mode == "eval" else "boxmot.engine.tuning.tuner")
    assert Path(captured["args"].build) == fixture.build
    assert captured["args"].cache_inputs is True
    assert captured["args"].evaluation_config["annotation_layout"] == "kitti_tracking"
    assert captured["result"].raw["car"]["HOTA"] == captured["result"].raw["pedestrian"]["HOTA"] == 100
