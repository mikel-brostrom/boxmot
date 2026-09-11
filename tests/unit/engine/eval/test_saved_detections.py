"""Saved boxes reach real image trackers and native KITTI 2D scoring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from boxmot.engine.eval import saved_detections
from boxmot.reid.protocols import EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec


class _Encoder:
    """Exercise real tracking without downloading appearance weights."""

    requirements = EncoderRequirements()
    embedding_dim = 2

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, frames, detections):
        self.calls += 1
        result = []
        for frame, batch in zip(frames, detections, strict=True):
            assert frame.image.shape == (3, 192, 192)
            assert batch.masks is None
            result.append(torch.nn.functional.one_hot(batch.class_ids - 1, num_classes=2).float())
        return result


def _dataset(root: Path, *, sequences: tuple[str, ...] = ("0000",)) -> Path:
    """Include an empty final frame and invalid mask RLEs that must stay unread."""
    pixels = np.random.default_rng(13).integers(0, 256, (192, 192, 3), dtype=np.uint8)
    for sequence in sequences:
        images = root / "images" / sequence
        images.mkdir(parents=True)
        detections, annotations = [], []
        for frame in range(3):
            assert cv2.imwrite(str(images / f"{frame:06d}.png"), pixels)
            if frame == 2:
                continue
            for identity, label, left in ((1, "Car", 10), (2, "Pedestrian", 100)):
                bounds = f"{left} 20 {left + 50} 80"
                detections.append(f"{frame} {bounds} 0.99 {identity} 192 192 not-decoded " + " ".join(["0"] * 128))
                annotations.append(f"{frame} {identity} {label} 0 0 -10 {bounds} -1 -1 -1 -1000 -1000 -1000 -10")
        (root / f"{sequence}-detections.txt").write_text("\n".join(detections) + "\n")
        (root / f"{sequence}-gt.txt").write_text("\n".join(annotations) + "\n")
    config = {
        "id": "saved-kitti",
        "format": {"layout": "sequence", "box_type": "aabb"},
        "storage": {"root": "."},
        "default_split": "val",
        "fps": 10,
        "modalities": {
            "images": {"format": "image-directory", "path": "images/{sequence}"},
            "detections_2d": {
                "format": "trackrcnn",
                "path": "{sequence}-detections.txt",
                "options": {"load_masks": False},
            },
            "ground_truth": {"format": "kitti-tracking-labels", "path": "{sequence}-gt.txt"},
        },
        "classes": {"target": {"car": 1, "pedestrian": 2}},
        "splits": {"val": {"partition": "training", "has_ground_truth": True, "sequences": list(sequences)}},
    }
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _args(root: Path, dataset: Path, **options) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "dataset": str(dataset),
            "tracker": "occluboost",
            "tracker_backend": "python",
            "tracker_config": None,
            "project": root / "runs",
            "split": "val",
            "sequence_names": (),
            "reid": "fixture-reid",
            "cache_inputs": False,
            **options,
        }
    )


def _stub_encoder(monkeypatch, encoder: _Encoder) -> None:
    monkeypatch.setattr(
        saved_detections,
        "resolve_reid_spec",
        lambda reference: (
            ReIDEncoderSpec("torch", artifact="fixture.pt", artifact_sha256="a" * 64),
            {"profile": reference},
        ),
    )
    monkeypatch.setattr(saved_detections, "create_reid_encoder", lambda spec: encoder)


@pytest.fixture(autouse=True)
def installed_trackeval() -> None:
    pytest.importorskip("trackeval")
    saved_detections.validate_trackeval_kitti_dependencies()


@pytest.mark.parametrize("per_class", (False, True))
def test_saved_boxes_use_real_occluboost_and_kitti_2d_metrics(monkeypatch, tmp_path: Path, per_class: bool) -> None:
    dataset = _dataset(tmp_path, sequences=("0000", "0001"))
    encoder = _Encoder()
    _stub_encoder(monkeypatch, encoder)
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, per_class=per_class))

    assert result.timings["frames"] == 6
    assert encoder.calls == 4
    for name in ("car", "pedestrian"):
        assert result.raw[name]["HOTA"] == result.raw[name]["MOTA"] == result.raw[name]["IDF1"] == 100
    for name in ("0000", "0001"):
        rows = (result.exp_dir / f"{name}.txt").read_text().splitlines()
        assert len(rows) == 4
        assert {int(row.split(",")[0]) for row in rows} == {1, 2}
    metadata = json.loads((result.exp_dir / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["per_class"] is per_class
    assert metadata["sequences"] == {"0000": 3, "0001": 3}
    assert metadata["reid"]["profile"] == "fixture-reid"
    protocol = json.loads((result.exp_dir / "evaluation.json").read_text())
    assert protocol["tracking"]["geometry"] == "2d"


def test_saved_boxes_can_disable_appearance_without_loading_reid(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    profile = tmp_path / "tracker.yaml"
    profile.write_text("use_embeddings: false\n")
    monkeypatch.setattr(saved_detections, "resolve_reid_spec", lambda reference: pytest.fail("ReID is disabled"))
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, reid=None, tracker_config=str(profile)))
    assert result.raw["car"]["HOTA"] == 100
    assert json.loads((result.exp_dir / "run.json").read_text())["reid"] is None


@pytest.mark.parametrize("unused", (False, True))
def test_saved_boxes_reject_missing_or_unused_reid_before_tracking(monkeypatch, tmp_path: Path, unused: bool) -> None:
    dataset = _dataset(tmp_path)
    profile = tmp_path / "tracker.yaml"
    profile.write_text(f"use_embeddings: {str(not unused).lower()}\n")
    monkeypatch.setattr(saved_detections, "resolve_reid_spec", lambda reference: pytest.fail("invalid ReID selection"))
    with pytest.raises(ValueError, match="does not use ReID" if unused else "requires appearance embeddings"):
        saved_detections.run_saved_detections(
            _args(tmp_path, dataset, tracker_config=str(profile), reid="fixture-reid" if unused else None)
        )
    assert not (tmp_path / "runs").exists()


def test_saved_boxes_honor_sequence_selection(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, sequences=("0000", "0001"))
    encoder = _Encoder()
    _stub_encoder(monkeypatch, encoder)
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, sequence_names=("0001",)))
    assert result.timings["frames"] == 3
    assert (result.exp_dir / "0001.txt").is_file()
    assert not (result.exp_dir / "0000.txt").exists()
    assert encoder.calls == 2


def test_saved_boxes_render_standard_evaluation_panel(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path)
    _stub_encoder(monkeypatch, _Encoder())
    result = saved_detections.main(_args(tmp_path, dataset))
    captured = capsys.readouterr()
    assert result.workflow_rendered
    assert "HOTA" in captured.out + captured.err
    assert "car" in captured.out + captured.err
    assert "Results:" in captured.out + captured.err


def test_saved_input_cache_reuses_boxes_images_features_and_rebuilds_changed_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = _dataset(tmp_path)
    encoder = _Encoder()
    _stub_encoder(monkeypatch, encoder)
    uncached = saved_detections.run_saved_detections(_args(tmp_path, dataset))
    initial = saved_detections.run_saved_detections(_args(tmp_path, dataset, cache_inputs=True))
    calls = encoder.calls
    assert calls == 4
    original = (initial.exp_dir / "0000.txt").read_bytes()
    assert original == (uncached.exp_dir / "0000.txt").read_bytes()
    assert initial.raw == uncached.raw
    repeated = saved_detections.run_saved_detections(_args(tmp_path, dataset, cache_inputs=True))
    assert encoder.calls == calls
    assert (repeated.exp_dir / "0000.txt").read_bytes() == original
    assert repeated.raw == initial.raw
    predictions = tmp_path / "0000-detections.txt"
    predictions.write_text(predictions.read_text().replace("10 20 60 80", "11 20 61 80", 1))
    saved_detections.run_saved_detections(_args(tmp_path, dataset, cache_inputs=True))
    assert encoder.calls == calls + 1
    pixels = np.zeros((192, 192, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "images/0000/000001.png"), pixels)
    saved_detections.run_saved_detections(_args(tmp_path, dataset, cache_inputs=True))
    assert encoder.calls == calls + 2


@pytest.mark.parametrize("tracker", ("botsort", "occluboost"))
def test_saved_boxes_support_native_trackers(monkeypatch, tmp_path: Path, tracker: str) -> None:
    dataset = _dataset(tmp_path)
    _stub_encoder(monkeypatch, _Encoder())
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, tracker=tracker, tracker_backend="cpp"))
    assert result.raw["car"]["HOTA"] == result.raw["pedestrian"]["HOTA"] == 100
    metadata = json.loads((result.exp_dir / "run.json").read_text())
    assert metadata["tracker_backend"] == "cpp"
    assert metadata["per_class"] is False


def test_saved_boxes_save_video_on_the_authored_timeline(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    _stub_encoder(monkeypatch, _Encoder())
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, save=True))
    video = result.exp_dir / "videos/0000.mp4"
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == 10
    finally:
        capture.release()
    assert result.args.video_paths == (video,)


@pytest.mark.parametrize("cache_inputs", (False, True))
def test_saved_boxes_skip_image_pixels_when_features_are_disabled(
    monkeypatch, tmp_path: Path, cache_inputs: bool
) -> None:
    from boxmot.datasets import sensor_cache

    dataset = _dataset(tmp_path)
    profile = tmp_path / "tracker.yaml"
    profile.write_text("use_embeddings: false\nuse_cmc: false\n")
    fail = lambda *args: pytest.fail("Disabled CMC/ReID must not load image pixels")
    monkeypatch.setattr(saved_detections, "read_rgb_chw_uint8", fail)
    monkeypatch.setattr(sensor_cache, "read_rgb_chw_uint8", fail)
    result = saved_detections.run_saved_detections(
        _args(tmp_path, dataset, reid=None, tracker_config=str(profile), cache_inputs=cache_inputs)
    )
    assert result.raw["car"]["HOTA"] == 100
