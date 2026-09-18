"""MOTS masks survive replay and serialize to strict official KITTI records."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot import create_tracker
from boxmot.datasets.masks import MASK_CODEC, pack_mask
from boxmot.datasets.schema import INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval import mots_io, replay
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Detections, MaskBatch, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


def _result(
    masks: torch.Tensor,
    *,
    indices: tuple[int, ...] = (0,),
    ids: tuple[int, ...] = (1,),
    scores: tuple[float, ...] = (0.9,),
    native_masks: torch.Tensor | None = None,
) -> PipelineResult:
    """Construct independent detection and track rows to exercise mask alignment."""
    detections = Detections(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 6.0, 4.0]] * len(masks))),
        scores=torch.ones(len(masks), dtype=torch.float32),
        class_ids=torch.ones(len(masks), dtype=torch.int64),
        sample_id="frame",
        masks=MaskBatch(masks),
    )
    tracks = Tracks(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 6.0, 4.0]] * len(ids))),
        track_ids=torch.tensor(ids),
        scores=torch.tensor(scores),
        class_ids=torch.ones(len(ids), dtype=torch.int64),
        detection_indices=torch.tensor(indices),
        sample_id="frame",
        masks=None if native_masks is None else MaskBatch(native_masks),
    )
    return PipelineResult(detections, tracks)


def test_box_tracks_follow_detection_indices_and_omit_unmatched_tracks() -> None:
    masks = torch.zeros((2, 4, 6), dtype=torch.bool)
    masks[0, 0, 0] = True
    masks[1, 3, 5] = True
    result = _result(masks, indices=(1, 0, -1), ids=(10, 20, 30), scores=(0.9, 0.9, 0.9))

    tracks = mots_io.prepare_mots_tracks(result, (4, 6))

    assert tracks.track_ids.tolist() == [10, 20]
    assert tracks.masks is not None
    torch.testing.assert_close(tracks.masks.values, masks.flip(0))
    assert tracks.masks.values.sum() == 2


def test_native_track_masks_take_precedence_and_keep_unmatched_native_track() -> None:
    detection_mask = torch.ones((1, 4, 6), dtype=torch.bool)
    native = torch.zeros_like(detection_mask)
    native[0, 1, 2] = True
    result = _result(detection_mask, indices=(-1,), native_masks=native)

    tracks = mots_io.prepare_mots_tracks(result, (4, 6))

    torch.testing.assert_close(tracks.masks.values, native)
    assert tracks.detection_indices.tolist() == [-1]


@pytest.mark.parametrize("scores,winning_id", [((0.4, 0.9), 20), ((0.9, 0.9), 10)])
def test_overlap_uses_score_then_smallest_id_and_discards_occluded_masks(scores, winning_id) -> None:
    masks = torch.ones((2, 4, 6), dtype=torch.bool)
    result = _result(masks, indices=(0, 1), ids=(10, 20), scores=scores)

    tracks = mots_io.prepare_mots_tracks(result, (4, 6))

    assert tracks.track_ids.tolist() == [winning_id]
    assert tracks.masks.values.all()
    assert result.detections.masks.values.all(), "Resolving overlap must not mutate cached masks."


def test_partial_overlap_is_removed_only_from_lower_score_mask() -> None:
    masks = torch.zeros((2, 4, 6), dtype=torch.bool)
    masks[0, :, :4] = True
    masks[1, :, 2:] = True
    result = _result(masks, indices=(0, 1), ids=(10, 20), scores=(0.9, 0.8))

    tracks = mots_io.prepare_mots_tracks(result, (4, 6))

    assert tracks.track_ids.tolist() == [10, 20]
    assert tracks.masks.values.sum(dim=0).max() == 1
    torch.testing.assert_close(tracks.masks.values[0], masks[0])
    assert tracks.masks.values[1].sum() == 8


def test_mots_rejects_wrong_dimensions_classes_and_missing_masks() -> None:
    result = _result(torch.ones((1, 4, 6), dtype=torch.bool))
    with pytest.raises(ValueError, match="frame size"):
        mots_io.prepare_mots_tracks(result, (5, 6))
    with pytest.raises(ValueError, match="class IDs"):
        mots_io.prepare_mots_tracks(replace(result, tracks=replace(result.tracks, class_ids=torch.tensor([0]))), (4, 6))
    with pytest.raises(ValueError, match="requires published detection masks"):
        mots_io.prepare_mots_tracks(replace(result, detections=replace(result.detections, masks=None)), (4, 6))


def test_mots_codec_round_trips_large_ids_irregular_masks_and_zero_based_frames(tmp_path: Path) -> None:
    api = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(42)
    mask = rng.random((17, 23)) > 0.5
    encoded = mots_io.encode_mots_mask(mask)
    rows = [mots_io.MOTSRow(0, 2**60 + 7, 2, encoded), mots_io.MOTSRow(7, 9, 1, encoded)]
    destination = tmp_path / "0000.txt"
    with destination.open("w") as handle:
        mots_io.write_mots_rows(handle, rows)

    loaded = mots_io.read_mots_results(destination, frame_shapes={0: (17, 23), 7: (17, 23)})

    assert loaded == {0: (rows[0],), 7: (rows[1],)}
    assert destination.read_text().startswith(f"0 {2**60 + 7} 2 17 23 ")
    np.testing.assert_array_equal(api.decode(loaded[0][0].rle), mask)


@pytest.mark.parametrize("counts", [b"", b"/", b"P", b"1", b"000", b"oooooooo0", b"o0"])
def test_compressed_runs_are_checked_before_native_decode(counts: bytes) -> None:
    with pytest.raises(ValueError, match="RLE"):
        mots_io.EncodedMask(4, 6, counts)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda text: text.replace("0 7 1", "-1 7 1"), "integers"),
        (lambda text: text.replace("0 7 1", "0 -7 1"), "integers"),
        (lambda text: text.replace("0 7 1", "0 7 3"), "class IDs"),
        (lambda text: text.replace("0 7 1", "1 7 1"), "Unknown MOTS frame"),
        (lambda text: text.replace("4 6", "5 6"), "dimensions"),
        (lambda text: text + text, "Duplicate MOTS track ID"),
        (lambda text: text + text.replace("0 7 1", "0 8 2"), "Overlapping MOTS"),
    ],
)
def test_reader_rejects_invalid_prediction_rows(tmp_path: Path, mutation, message: str) -> None:
    pytest.importorskip("pycocotools.mask")
    handle = io.StringIO()
    mots_io.write_mots_rows(handle, [mots_io.MOTSRow(0, 7, 1, mots_io.encode_mots_mask(np.ones((4, 6), bool)))])
    path = tmp_path / "0000.txt"
    path.write_text(mutation(handle.getvalue()))

    with pytest.raises(ValueError, match=message):
        mots_io.read_mots_results(path, frame_shapes={0: (4, 6)})


def test_missing_optional_codec_reports_install_command(monkeypatch) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("No module named 'pycocotools'", name="pycocotools")

    monkeypatch.setattr(mots_io.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="uv sync --extra cpu --extra mots"):
        mots_io.encode_mots_mask(np.ones((4, 6), bool))


def _mots_build(tmp_path: Path, *, publish_masks: bool = True) -> Path:
    """Publish two sequences containing a non-rectangular mask and an empty frame."""
    detect = StagePlan.create("detect", component={"id": "mots-fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="mots-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("mots-fixture"),
        publish=PublishOptions(image_references=True, masks=publish_masks),
        stages=(detect, finalize),
        metadata={"split": "validation"},
    )
    image_dir = plan.staging_root / "images"
    image_dir.mkdir(parents=True)
    samples, instances, masks = [], [], []
    for sequence, frame_index in (("0000", 0), ("0000", 1), ("0001", 0)):
        sample_id = f"{sequence}-{frame_index}"
        instance_id = f"{plan.build_id}:{sample_id}:0"
        assert cv2.imwrite(str(image_dir / f"{sample_id}.png"), np.zeros((4, 6, 3), np.uint8))
        samples.append(
            dict(
                sample_id=sample_id,
                split="validation",
                sequence_id=sequence,
                frame_index=frame_index,
                timestamp_s=None,
                image_ref=f"images/{sample_id}.png",
                height=4,
                width=6,
            )
        )
        if frame_index:
            continue
        instances.append(
            dict(
                sample_id=sample_id,
                instance_id=instance_id,
                detection_index=0,
                x1=0.0,
                y1=0.0,
                x2=6.0,
                y2=4.0,
                score=0.95,
                class_id=1,
            )
        )
        mask = torch.eye(4, 6, dtype=torch.bool)
        masks.append(
            dict(
                sample_id=sample_id,
                instance_id=instance_id,
                height=4,
                width=6,
                codec=MASK_CODEC,
                data=pack_mask(mask),
            )
        )
    writer = ParquetShardWriter(plan.staging_root, box_type=plan.box_type)
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    if publish_masks:
        writer.write(MASKS_ARTIFACT, masks, shard_index=0)
    return finalize_build(plan)


def test_mots_replay_preserves_masks_in_injected_spawned_and_callback_paths(tmp_path: Path) -> None:
    api = pytest.importorskip("pycocotools.mask")
    build = _mots_build(tmp_path)
    spec = TrackerSpec(name="bytetrack")
    tracker = create_tracker(spec)
    assert not tracker.requirements.masks
    captured = []
    results = []
    for name, kwargs in (
        ("injected", {"tracker": tracker}),
        ("spawned", {"workers": 2}),
        ("callback", {"frame_callback": captured.append}),
    ):
        result = replay.replay_build(build, spec, output_dir=tmp_path / name, output_format="mots", **kwargs)
        assert result.frames == 3
        assert result.track_rows == 2
        assert [path.name for path in result.sequence_files] == ["0000.txt", "0001.txt"]
        assert not list(result.output_dir.glob(".replay-*"))
        results.append([path.read_bytes() for path in result.sequence_files])
        for path in result.sequence_files:
            rows = mots_io.read_mots_results(path, frame_shapes={0: (4, 6), 1: (4, 6)})
            assert list(rows) == [0]
            np.testing.assert_array_equal(api.decode(rows[0][0].rle), np.eye(4, 6, dtype=bool))
    assert results[0] == results[1] == results[2]
    assert len(captured) == 3
    assert [len(frame.result.tracks) for frame in captured] == [1, 0, 1]
    for frame in captured:
        assert frame.sample.frame is not None
        assert frame.result.tracks.masks is not None
        if len(frame.result.tracks):
            torch.testing.assert_close(frame.result.tracks.masks.values[0], torch.eye(4, 6, dtype=torch.bool))


@pytest.mark.parametrize("callback", [False, True])
def test_mots_replay_requires_published_masks_for_box_trackers(tmp_path: Path, callback: bool) -> None:
    build = _mots_build(tmp_path, publish_masks=False)
    spec = TrackerSpec(name="bytetrack")
    with pytest.raises(ValueError, match="masks"):
        replay.replay_build(
            build,
            spec,
            output_dir=tmp_path / "outputs",
            tracker=create_tracker(spec),
            output_format="mots",
            frame_callback=(lambda _frame: None) if callback else None,
        )


def test_injected_mots_failure_preserves_existing_results(tmp_path: Path) -> None:
    pytest.importorskip("pycocotools.mask")
    build = _mots_build(tmp_path)
    spec = TrackerSpec(name="bytetrack")
    base = create_tracker(spec)

    class FailingTracker:
        name = "failing"
        supports_obb = False
        requirements = TrackerRequirements()

        def reset(self) -> None:
            base.reset()

        def update(self, detections: Detections, frame=None) -> Tracks:
            if detections.sample_id == "0000-1":
                raise RuntimeError("tracking failed")
            return base.update(detections=detections, frame=frame)

    output = tmp_path / "output"
    output.mkdir()
    (output / "0000.txt").write_text("original result")
    with pytest.raises(RuntimeError, match="tracking failed"):
        replay.replay_build(build, spec, output_dir=output, tracker=FailingTracker(), output_format="mots")
    assert (output / "0000.txt").read_text() == "original result"
    assert not (output / "0001.txt").exists()
    assert not list(output.glob(".replay-*"))


def test_unknown_output_format_fails_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="output_format"):
        replay.replay_build(tmp_path, TrackerSpec(name="bytetrack"), output_dir=output, output_format="invalid")
    assert not output.exists()
