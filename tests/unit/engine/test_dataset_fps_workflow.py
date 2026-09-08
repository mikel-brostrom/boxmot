"""Regression coverage for dataset FPS across perception, replay, and metrics."""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch

import boxmot.engine.eval.catalog_cache as catalog_cache
import boxmot.engine.eval.evaluator as evaluator
import boxmot.engine.materialization.stages.detect as detect_stage
import boxmot.engine.materialization.stages.embed as embed_stage
import boxmot.engine.materialization.workflow as workflow
from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.engine.eval.replay import iter_cached_tracks, tracks_to_mot_rows
from boxmot.engine.materialization.catalog import catalog_mot_dataset
from boxmot.reid import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, Tracks
from boxmot.trackers import TrackerRequirements


class _PixelDetector:
    """Recover original frame numbers from pixels, including an empty frame."""

    capabilities = DetectorCapabilities()

    def __init__(self) -> None:
        self.seen: list[int] = []
        self.provides_masks = False

    def predict(self, frames: list[Frame]) -> list[Detections]:
        outputs = []
        for frame in frames:
            original_frame = int(frame.image[0, 0, 0])
            self.seen.append(original_frame)
            count = int(original_frame != 7)
            outputs.append(
                Detections(
                    sample_id=frame.sample_id,
                    geometry=Boxes(torch.tensor([[1.0, 2.0, 4.0, 6.0]])[:count]),
                    scores=torch.full((count,), 0.95, dtype=torch.float32),
                    class_ids=torch.full((count,), 2, dtype=torch.int64),
                    masks=(
                        MaskBatch(torch.ones((count, *frame.image_size), dtype=torch.bool))
                        if self.provides_masks
                        else None
                    ),
                )
            )
        return outputs


class _PixelEncoder:
    """Make frame-specific embeddings to expose incorrectly joined caches."""

    embedding_dim = 2

    def __init__(self) -> None:
        self.seen: list[int] = []

    def encode(self, frames: list[Frame], detections: list[Detections]) -> list[torch.Tensor]:
        outputs = []
        for frame, batch in zip(frames, detections, strict=True):
            original_frame = int(frame.image[0, 0, 0])
            self.seen.append(original_frame)
            outputs.append(torch.tensor([[original_frame, 1.0]], dtype=torch.float32).repeat(len(batch), 1))
        return outputs


class _RecordingTracker:
    """Track the fixture's one object while recording the replayed inputs."""

    name = "fixture"
    supports_obb = False
    requirements = TrackerRequirements(embeddings=True, frame=True)

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
        assert frame is not None
        self.frames.append(frame)
        assert detections.embeddings is not None
        if len(detections):
            assert detections.embeddings[:, 0].tolist() == [float(frame.image[0, 0, 0])]
        return Tracks(
            sample_id=detections.sample_id,
            geometry=detections.geometry,
            track_ids=torch.ones(len(detections), dtype=torch.int64),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.arange(len(detections), dtype=torch.int64),
        )


@pytest.fixture
def fps_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Create an actual MOT sequence and inject only model/config resolution."""

    data_root = tmp_path / "datasets"
    split_root = data_root / "fixture" / "ablation"
    sequence = split_root / "sequence"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    for original_frame in range(1, 20):
        pixels = np.full((8, 10, 3), original_frame, dtype=np.uint8)
        assert cv2.imwrite(str(sequence / "img1" / f"{original_frame:06d}.png"), pixels)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=sequence\nimDir=img1\nframeRate=30\nseqLength=19\nimWidth=10\nimHeight=8\nimExt=.png\n",
        encoding="utf-8",
    )
    gt_path = sequence / "gt" / "gt.txt"
    gt_path.write_text(
        "".join(f"{frame},1,1,2,3,4,1,7,1\n" for frame in range(1, 20) if frame != 7),
        encoding="utf-8",
    )
    dataset = {
        "id": "fixture",
        "root": "fixture",
        "split": "ablation",
        "default_split": "ablation",
        "layout": "mot",
        "box_type": "aabb",
        "splits": {"ablation": {"path": "ablation", "annotations": None, "has_ground_truth": True}},
        "classes": {"vehicle": {"id": 7, "evaluation": "target"}},
    }
    experiment = {
        "id": "fixture-ablation-detector",
        "source_path": tmp_path / "fixture-experiment.yaml",
        "dataset": dataset,
        "detector": {"ref": "fixture", "checkpoint": "default"},
        "segmentor": None,
        "reid": {"model": "fixture-reid.pt"},
        "evaluation": {"classes": [{"name": "vehicle", "dataset_id": 7, "detector_name": "car", "detector_id": 2}]},
    }
    detector = _PixelDetector()
    encoder = _PixelEncoder()
    detector_spec = DetectorSpec("fixture", geometry_mode="aabb", device="cpu")
    encoder_spec = ReIDEncoderSpec("fixture", device="cpu", options=(("embedding_dim", 2),))
    detector_provenance = {"spec": {"backend": "fixture", "device": "cpu"}}
    encoder_provenance = {"spec": {"backend": "fixture", "device": "cpu"}}
    for module in (workflow, evaluator):
        monkeypatch.setattr(module, "resolve_experiment_config", lambda *_args, **_kwargs: experiment)
        monkeypatch.setattr(
            module,
            "resolve_detector_spec",
            lambda *_args, **_kwargs: (detector_spec, detector_provenance),
        )
        monkeypatch.setattr(
            module,
            "resolve_reid_spec",
            lambda *_args, **_kwargs: (encoder_spec, encoder_provenance),
        )
    monkeypatch.setattr(evaluator, "load_dataset_config", lambda _reference: dict(dataset))
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities())
    monkeypatch.setattr(detect_stage, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(embed_stage, "_WORKER_ENCODERS", {})
    monkeypatch.setattr(detect_stage, "create_detector", lambda _spec: detector)
    monkeypatch.setattr(embed_stage, "create_reid_encoder", lambda _spec: encoder)
    for module in (workflow, catalog_cache):
        monkeypatch.setattr(
            module,
            "default_source_metadata_cache_path",
            lambda _source: tmp_path / "source-metadata.json",
        )

    return SimpleNamespace(
        root=tmp_path,
        data_root=data_root,
        split_root=split_root,
        dataset=dataset,
        experiment=experiment,
        gt_path=gt_path,
        detector=detector,
        encoder=encoder,
        detector_provenance=detector_provenance,
        encoder_provenance=encoder_provenance,
    )


def _materialize(case: SimpleNamespace, fps: float | None, **overrides: object) -> Path:
    """Execute the real catalog, detector, encoder, and Parquet build stages."""

    options = {
        "experiment": case.experiment["source_path"],
        "data_root": case.data_root,
        "fps": fps,
        "device": "cpu",
        "materialize_explicit_keys": (),
        "build_root": case.root / "builds",
        "plan_overrides": ("decode.workers=1", "detect.batch_size=3", "embed.batch_size=2"),
        "publish_image_refs": True,
        "publish_masks": False,
        "publish_embeddings": True,
        **overrides,
    }
    return workflow.materialize(SimpleNamespace(**options))


def _eval_args(case: SimpleNamespace, build: Path, fps: float | None) -> argparse.Namespace:
    """Construct the dataset-backed evaluator arguments used by the CLI."""

    return argparse.Namespace(
        dataset="fixture",
        split="ablation",
        data_root=case.data_root,
        build=build,
        fps=fps,
        project=case.root / "runs",
        exp_dir=case.root / "results",
    )


@pytest.mark.parametrize(("fps", "reuse_native"), [(None, False), (5.0, False), (5.0, True)])
def test_dataset_fps_keeps_perception_replay_and_ground_truth_aligned(
    fps_case: SimpleNamespace, fps: float | None, reuse_native: bool
) -> None:
    """Original pixels/times, keyed embeddings, dense frames, and GT agree."""

    original_gt = fps_case.gt_path.read_bytes()
    if reuse_native:
        _materialize(fps_case, None)
        fps_case.detector.seen.clear()
        fps_case.encoder.seen.clear()
    build = _materialize(fps_case, fps)
    expected_frames = list(range(1, 20)) if fps is None else [1, 7, 13, 19]
    expected_nonempty = [frame for frame in expected_frames if frame != 7]
    assert fps_case.detector.seen == ([] if reuse_native else expected_frames)
    assert fps_case.encoder.seen == ([] if reuse_native else expected_nonempty)

    manifest = DatasetManifest.load(build)
    assert manifest.metadata.get("fps") == fps
    dataset = CachedVisionDataset(build, load_images=True, load_embeddings=True)
    assert len(dataset) == len(expected_frames)
    assert [sample.frame_index for sample in dataset] == list(range(len(expected_frames)))
    assert [sample.timestamp_s for sample in dataset] == pytest.approx([(frame - 1) / 30 for frame in expected_frames])
    assert [int(sample.frame.image[0, 0, 0]) for sample in dataset] == expected_frames
    for original_frame, sample in zip(expected_frames, dataset, strict=True):
        assert sample.image_ref.endswith(f"{original_frame:06d}.png")
        assert len(sample.detections) == int(original_frame != 7)
        assert sample.detections.embeddings.shape == (len(sample.detections), 2)
        if len(sample.detections):
            assert sample.detections.class_ids.tolist() == [7]
            assert sample.detections.instance_ids == (f"{manifest.build_id}:{sample.sample_id}:0",)
            assert sample.detections.embeddings.tolist() == [[float(original_frame), 1.0]]

    args = _eval_args(fps_case, build, fps)
    evaluator.eval_setup(args)
    assert args.seq_info == {"sequence": len(expected_frames)}
    assert args.fps == fps
    aligned_gt = np.loadtxt(args.gt_folder / "sequence" / "gt" / "gt.txt", delimiter=",")
    expected_output_frames = [index + 1 for index, frame in enumerate(expected_frames) if frame != 7]
    assert aligned_gt[:, 0].tolist() == expected_output_frames
    assert fps_case.gt_path.read_bytes() == original_gt
    if fps is None:
        assert args.source == fps_case.split_root
        assert args.gt_folder == fps_case.split_root
        assert "frame_sampling" not in manifest.metadata
    else:
        assert manifest.metadata["frame_sampling"] == {"sequence": tuple(expected_frames)}
        info = configparser.ConfigParser()
        info.read(args.source / "sequence" / "seqinfo.ini")
        assert info["Sequence"].getfloat("frameRate") == fps
        assert info["Sequence"].getint("seqLength") == len(expected_frames)
        sampled_images = sorted((args.source / "sequence" / "img1").glob("*.png"))
        assert [int(cv2.imread(str(image))[0, 0, 0]) for image in sampled_images] == expected_frames

    tracker = _RecordingTracker()
    replayed = list(iter_cached_tracks(dataset, tracker))
    assert tracker.resets == 1
    assert [int(frame.image[0, 0, 0]) for frame in tracker.frames] == expected_frames
    assert [frame.timestamp_s for frame in tracker.frames] == pytest.approx(
        [(frame - 1) / 30 for frame in expected_frames]
    )
    rows = [row for item in replayed for row in tracks_to_mot_rows(item.result.tracks, item.sample.frame_index)]
    args.exp_dir.mkdir()
    np.savetxt(args.exp_dir / "sequence.txt", np.asarray(rows), delimiter=",")
    results = evaluator.run_motmetrics(args, verbose=False)
    assert results["vehicle"]["HOTA"] == pytest.approx(100.0)
    assert results["vehicle"]["MOTA"] == pytest.approx(100.0)
    assert results["vehicle"]["IDF1"] == pytest.approx(100.0)
    assert results["vehicle"]["Frames"] == len(expected_frames)
    assert results["vehicle"]["GT_Dets"] == len(expected_nonempty)
    assert results["vehicle"]["Dets"] == len(expected_nonempty)


def test_dataset_fps_build_identity_and_reuse(fps_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct FPS builds reuse perception while retaining their own identity."""

    native = _materialize(fps_case, None)
    detector_calls = list(fps_case.detector.seen)
    encoder_calls = list(fps_case.encoder.seen)
    monkeypatch.setattr(detect_stage, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(embed_stage, "_WORKER_ENCODERS", {})
    detector_factory = Mock(return_value=fps_case.detector)
    encoder_factory = Mock(return_value=fps_case.encoder)
    monkeypatch.setattr(detect_stage, "create_detector", detector_factory)
    monkeypatch.setattr(embed_stage, "create_reid_encoder", encoder_factory)
    five = _materialize(fps_case, 5.0)
    ten = _materialize(fps_case, 10.0)
    assert len({native, five, ten}) == 3
    assert [len(CachedVisionDataset(build)) for build in (native, five, ten)] == [19, 4, 7]
    assert fps_case.detector.seen == detector_calls
    assert fps_case.encoder.seen == encoder_calls
    assert _materialize(fps_case, 5.0) == five
    assert fps_case.detector.seen == detector_calls
    assert fps_case.encoder.seen == encoder_calls
    detector_factory.assert_not_called()
    encoder_factory.assert_not_called()
    native_catalog = catalog_mot_dataset(fps_case.dataset, split="ablation", data_root=fps_case.data_root)
    assert DatasetManifest.load(native).metadata["source_catalog_digest"] == native_catalog.fingerprint


def test_reused_fps_build_has_canonical_identity_and_experiment_provenance(fps_case: SimpleNamespace) -> None:
    """Cache reuse preserves independent-build keys and authored eval checks."""

    _materialize(fps_case, None)
    fps_case.detector.seen.clear()
    fps_case.encoder.seen.clear()
    reused = _materialize(fps_case, 5.0)
    assert fps_case.detector.seen == []
    assert fps_case.encoder.seen == []
    inferred = _materialize(fps_case, 5.0, build_root=fps_case.root / "independent-builds")
    assert fps_case.detector.seen == [1, 7, 13, 19]
    assert fps_case.encoder.seen == [1, 13, 19]
    reused_manifest = DatasetManifest.load(reused)
    inferred_manifest = DatasetManifest.load(inferred)
    assert reused_manifest.build_id == inferred_manifest.build_id
    for key in ("source_catalog_digest", "class_bridge", "components", "component_fingerprints", "experiment_id"):
        assert reused_manifest.metadata[key] == inferred_manifest.metadata[key]
    reused_dataset = CachedVisionDataset(reused, load_embeddings=True)
    inferred_dataset = CachedVisionDataset(inferred, load_embeddings=True)
    for cached, fresh in zip(reused_dataset, inferred_dataset, strict=True):
        assert cached.sample_id == fresh.sample_id
        assert cached.detections.instance_ids == fresh.detections.instance_ids
        assert torch.equal(cached.detections.geometry.values, fresh.detections.geometry.values)
        assert torch.equal(cached.detections.embeddings, fresh.detections.embeddings)
        assert cached.frame_index == fresh.frame_index
        assert cached.timestamp_s == fresh.timestamp_s
    args = _eval_args(fps_case, reused, 5.0)
    args.dataset = None
    args.experiment = fps_case.experiment["source_path"]
    evaluator.eval_setup(args)
    assert args.experiment_id == fps_case.experiment["id"]
    assert args.seq_info == {"sequence": 4}


@pytest.mark.parametrize("incompatible", ["detector", "reid", "source", "class_mapping", "embeddings"])
def test_dataset_fps_does_not_reuse_incompatible_parent(fps_case: SimpleNamespace, incompatible: str) -> None:
    """Changed perception, source content, class IDs, or payloads need new work."""

    _materialize(fps_case, None, publish_embeddings=incompatible != "embeddings")
    fps_case.detector.seen.clear()
    fps_case.encoder.seen.clear()
    if incompatible == "detector":
        fps_case.detector_provenance["spec"]["backend"] = "other-detector"
    elif incompatible == "reid":
        fps_case.encoder_provenance["spec"]["backend"] = "other-reid"
    elif incompatible == "source":
        image_path = fps_case.split_root / "sequence" / "img1" / "000013.png"
        assert cv2.imwrite(str(image_path), np.full((8, 10, 3), 42, dtype=np.uint8))
    elif incompatible == "class_mapping":
        fps_case.experiment["evaluation"]["classes"][0]["dataset_id"] = 9

    build = _materialize(fps_case, 5.0)

    expected_nonempty = [1, 42, 19] if incompatible == "source" else [1, 13, 19]
    assert fps_case.encoder.seen == expected_nonempty
    if incompatible in {"detector", "source", "class_mapping"}:
        assert fps_case.detector.seen == [1, 7, expected_nonempty[1], 19]
    if incompatible == "class_mapping":
        assert CachedVisionDataset(build)[0].detections.class_ids.tolist() == [9]


def test_dataset_fps_does_not_reuse_parent_missing_requested_masks(
    fps_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmasked parent cannot satisfy publication of native detector masks."""

    fps_case.detector.provides_masks = True
    monkeypatch.setattr(workflow, "detector_capabilities", lambda _spec: DetectorCapabilities(provides_masks=True))
    _materialize(fps_case, None, publish_masks=False)
    fps_case.detector.seen.clear()
    fps_case.encoder.seen.clear()

    build = _materialize(fps_case, 5.0, publish_masks=True)

    assert fps_case.detector.seen == [1, 7, 13, 19]
    assert fps_case.encoder.seen == [1, 13, 19]
    dataset = CachedVisionDataset(build, load_masks=True)
    assert dataset[0].detections.masks is not None
    assert dataset[1].detections.masks.values.shape == (0, 8, 10)


def test_evaluation_infers_fps_from_explicit_build(fps_case: SimpleNamespace) -> None:
    """An explicit sampled build selects its own FPS when the flag is absent."""

    build = _materialize(fps_case, 5.0)
    args = _eval_args(fps_case, build, None)

    evaluator.eval_setup(args)

    assert args.fps == 5.0
    assert args.seq_info == {"sequence": 4}
    gt = np.loadtxt(args.gt_folder / "sequence" / "gt" / "gt.txt", delimiter=",")
    assert gt[:, 0].tolist() == [1, 3, 4]


@pytest.mark.parametrize(("build_fps", "requested_fps"), [(None, 5.0), (5.0, 10.0)])
def test_evaluation_rejects_mismatched_explicit_build_fps(
    fps_case: SimpleNamespace, build_fps: float | None, requested_fps: float
) -> None:
    """Conflicting FPS fails before replay can compare incompatible frames."""

    build = _materialize(fps_case, build_fps)
    args = _eval_args(fps_case, build, requested_fps)

    with pytest.raises(ValueError, match="FPS.*requested"):
        evaluator.eval_setup(args)
