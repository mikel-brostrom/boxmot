"""Propagate incoming workflow frames without binding guidance to source files."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from boxmot.detectors import DetectorCapabilities
from boxmot.engine.cli import boxmot
from boxmot.engine.tracking import workflow
from boxmot.engine.tracking.sinks import NullSink
from boxmot.structures import Boxes, Detections, Frame, MaskBatch
from boxmot.trackers.common.mask_guidance import MaskGuidance, MaskGuidanceConfig
from tests.unit.engine.eval.test_visualization import capture_rendered_images

BOX_TRACKERS = (
    "boosttrack",
    "botsort",
    "bytetrack",
    "deepocsort",
    "hybridsort",
    "occluboost",
    "ocsort",
    "sfsort",
    "strongsort",
)


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    """Provide a camera source and checkpoint without loading any model."""
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    values = dict(
        tracker="bytetrack",
        tracker_backend="python",
        geometry="aabb",
        per_class=False,
        classes=None,
        asso_func=None,
        source="0",
        edgetam=True,
        mask_guidance_weights=checkpoint,
        vid_stride=1,
        device="cpu",
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
def test_track_cli_passes_mask_guidance_separately_from_tracker_options(
    monkeypatch, tmp_path: Path, tracker_name
) -> None:
    args = _args(tmp_path, tracker=tracker_name, asso_func="iou")
    received = []
    monkeypatch.setattr(workflow, "main", received.append)

    result = CliRunner().invoke(
        boxmot,
        [
            "track",
            "--edgetam",
            "--tracker",
            tracker_name,
            "--asso-func",
            "iou",
            "--source",
            args.source,
            "--mask-guidance-weights",
            str(args.mask_guidance_weights),
            "--mask-guidance-max-objects",
            "4",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received[0].edgetam is True
    assert received[0].mask_guidance_weights == args.mask_guidance_weights
    assert received[0].mask_guidance_max_objects == 4
    assert "mask_guidance_weights" not in workflow._tracker_spec(received[0], "aabb").option_dict
    assert "--mask-guidance-weights" in CliRunner().invoke(boxmot, ["tune", "--help"]).output
    materialize_help = CliRunner().invoke(boxmot, ["materialize", "--help"]).output
    assert "--mask-guidance-weights" not in materialize_help


@pytest.mark.parametrize("checkpoint", ("edgetam.pt", "models/edgetam.pt"))
def test_track_cli_accepts_checkpoint_to_download(monkeypatch, tmp_path: Path, checkpoint: str) -> None:
    """A standard model name can reach the workflow before its weights exist."""
    monkeypatch.chdir(tmp_path)
    received = []
    monkeypatch.setattr(workflow, "main", received.append)

    result = CliRunner().invoke(
        boxmot,
        [
            "track",
            "--edgetam",
            "--tracker",
            "bytetrack",
            "--source",
            "0",
            "--mask-guidance-weights",
            checkpoint,
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0].mask_guidance_weights == Path(checkpoint)
    assert not Path(checkpoint).exists()


@pytest.mark.parametrize(
    "flags, enabled",
    [([], False), (["--edgetam"], True), (["--no-edgetam"], False), (["--edgetam", "--no-edgetam"], False)],
)
def test_track_flag_controls_guidance_independently_of_checkpoint(monkeypatch, tmp_path, flags, enabled) -> None:
    """Keep the selected checkpoint while enabling or disabling temporal inference."""
    from boxmot.engine.config.trackers import edgetam_checkpoint

    received = []
    checkpoint = tmp_path / "custom.pt"
    monkeypatch.setattr(workflow, "main", received.append)
    result = CliRunner().invoke(
        boxmot,
        ["track", "--source", "0", "--tracker", "bytetrack", "--device", "cpu",
         "--mask-guidance-weights", str(checkpoint), *flags],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    args = received[0]
    assert args.edgetam is enabled
    assert args.mask_guidance_weights == checkpoint
    assert edgetam_checkpoint(args) == (checkpoint if enabled else None)
    assert not checkpoint.exists()


@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
def test_disabled_guidance_skips_checkpoint_resolution_and_component_creation(monkeypatch, tmp_path, tracker_name):
    """An unchanged weights selection must not activate the model when toggled off."""
    from boxmot.segmentors.propagation import weights

    monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", lambda *_args: pytest.fail("Resolved disabled weights"))

    class CreatedTracker(Exception):
        """Stop after observing ordinary tracker construction."""

    def create_tracker(spec, **kwargs):
        assert spec.name == tracker_name
        assert kwargs == {}
        raise CreatedTracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)
    args = _args(tmp_path, tracker=tracker_name, edgetam=False, mask_guidance_weights=tmp_path / "missing.pt")
    with pytest.raises(CreatedTracker):
        workflow.run_track(args, detector=object())


def test_edgetam_flag_uses_standard_checkpoint_when_no_weights_are_selected(monkeypatch, tmp_path):
    from boxmot.segmentors.propagation import weights

    args = _args(tmp_path, mask_guidance_weights=None)
    resolved = tmp_path / "edgetam.pt"
    requests = []

    def resolve(checkpoint):
        requests.append(checkpoint)
        return resolved

    monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", resolve)
    config = workflow._mask_guidance_config(args, workflow._tracker_spec(args, "aabb"))

    assert config.checkpoint == resolved
    assert requests == [Path("edgetam.pt")]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tracker": "maf_hda"}, "Python 2D box tracker"),
        ({"tracker": "eagermot"}, "requires 3D detections and a CameraModel"),
        ({"tracker_backend": "cpp"}, "Python 2D box tracker"),
        ({"geometry": "obb"}, "requires AABB"),
        ({"tracker": "hybridsort"}, "asso_func='iou'"),
        ({"per_class": True}, "per_class=False"),
        ({"asso_func": "giou"}, "asso_func='iou'"),
    ],
)
def test_invalid_guidance_runs_fail_before_model_loading(monkeypatch, tmp_path, overrides, message) -> None:
    from boxmot.segmentors.propagation import weights

    monkeypatch.setattr(workflow, "create_detector", lambda *_args: pytest.fail("Loaded a detector"))
    monkeypatch.setattr(
        weights, "resolve_edgetam_checkpoint", lambda *_args: pytest.fail("Resolved invalid-run weights")
    )
    with pytest.raises(ValueError, match=message):
        workflow.run_track(_args(tmp_path, **overrides))


@pytest.mark.parametrize("source", ("0", "rtsp://camera.test/live", "https://camera.test/live", "*.png"))
@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
def test_workflow_configures_guidance_independently_of_source(
    monkeypatch, tmp_path: Path, source: str, tracker_name
) -> None:
    args = _args(tmp_path, source=source, vid_stride=2, tracker=tracker_name, asso_func="iou")
    received = []

    class CreatedTracker(Exception):
        """Stop immediately after observing tracker factory arguments."""

    def create_tracker(spec, *, mask_guidance):
        received.append((spec, mask_guidance))
        raise CreatedTracker

    monkeypatch.setattr(workflow, "create_tracker", create_tracker)
    with pytest.raises(CreatedTracker):
        workflow.run_track(args, detector=object())

    spec, config = received[0]
    assert spec.name == tracker_name
    assert config.checkpoint == args.mask_guidance_weights.resolve()
    assert config.device == "cpu"
    assert config.max_objects == 32


def test_workflow_resolves_guidance_checkpoint_before_creating_tracker(monkeypatch, tmp_path: Path) -> None:
    """The tracker receives the downloaded model path, independent of its source."""
    from boxmot.segmentors.propagation import weights

    monkeypatch.chdir(tmp_path)
    resolved = tmp_path / "models" / "edgetam.pt"
    requests = []

    def resolve(checkpoint: str | Path) -> Path:
        requests.append(checkpoint)
        resolved.parent.mkdir()
        resolved.write_bytes(b"downloaded model")
        return resolved

    class CreatedTracker(Exception):
        """Stop before loading a detector or opening a webcam."""

    def create_tracker(spec, *, mask_guidance):
        assert spec.name == "bytetrack"
        assert mask_guidance.checkpoint == resolved
        assert mask_guidance.checkpoint.read_bytes() == b"downloaded model"
        raise CreatedTracker

    monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", resolve)
    monkeypatch.setattr(workflow, "create_tracker", create_tracker)
    args = _args(tmp_path)
    args.mask_guidance_weights.unlink()
    args.mask_guidance_weights = Path("edgetam.pt")

    with pytest.raises(CreatedTracker):
        workflow.run_track(args, detector=object())

    assert requests == [Path("edgetam.pt")]


def test_guidance_workflow_rejects_overriding_an_injected_tracker(tmp_path) -> None:
    with pytest.raises(ValueError, match="Configure mask guidance on the supplied tracker"):
        workflow.run_track(_args(tmp_path), tracker=object())


@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
@pytest.mark.parametrize("cap_override", [None, 16])
def test_workflow_uses_mask_settings_from_tracker_profile(
    tmp_path: Path, tracker_name: str, cap_override: int | None
) -> None:
    """Saved tuning profiles control the effective guidance config before model loading."""
    profile = tmp_path / "best.yaml"
    profile.write_text(
        "edgetam:\n"
        "  min_coverage: 0.75\n"
        "  min_fill: 0.12\n"
        "  prompt_overlap: 0.25\n"
        "  max_objects: 8\n"
    )
    args = _args(
        tmp_path, tracker=tracker_name, tracker_config=str(profile), asso_func="iou",
        mask_guidance_max_objects=cap_override,
    )
    spec = workflow._tracker_spec(args, "aabb")

    config = workflow._mask_guidance_config(args, spec)

    assert config.min_coverage == 0.75
    assert config.min_fill == 0.12
    assert config.prompt_overlap == 0.25
    assert config.max_objects == (8 if cap_override is None else cap_override)
    assert spec.option_dict["edgetam.max_objects"] == config.max_objects


@pytest.mark.parametrize("source_indices", ((0, 1, 2), (100, 102, 106)))
def test_guidance_consumes_injected_stream_frames_as_they_arrive(monkeypatch, tmp_path: Path, source_indices) -> None:
    """A source can require every current frame to finish before producing another."""
    received = []

    class StreamSource:
        closed = False

        def __iter__(self) -> Iterator[Frame]:
            for index, source_index in enumerate(source_indices):
                assert received == list(range(index))
                yield Frame(
                    image=torch.full((3, 4, 5), index, dtype=torch.uint8),
                    sample_id=f"live:{index}",
                    sequence_id="live",
                    frame_index=source_index,
                    source_uri="rtsp://camera.test/live",
                )

        def close(self) -> None:
            self.closed = True

    class EmptyDetector:
        capabilities = DetectorCapabilities()

        def predict(self, frames) -> list[Detections]:
            return [
                Detections(
                    geometry=Boxes(torch.empty((0, 4), dtype=torch.float32)),
                    scores=torch.empty(0, dtype=torch.float32),
                    class_ids=torch.empty(0, dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
                for frame in frames
            ]

    def advance(_guidance, frame_index: int, frame: np.ndarray) -> None:
        assert frame_index == len(received)
        received.append(int(frame[0, 0, 0]))

    monkeypatch.setattr(MaskGuidance, "advance", advance)
    monkeypatch.setattr(workflow, "create_frame_source", lambda *_args, **_kwargs: pytest.fail("Reopened stream"))
    source = StreamSource()

    result = workflow.run_track(
        _args(tmp_path),
        detector=EmptyDetector(),
        source=source,
        sinks=(NullSink(),),
    )

    assert received == [0, 1, 2]
    assert result.summary.frames == 3
    assert source.closed


@pytest.mark.parametrize("command", ["track", "eval"])
def test_guidance_cli_rejects_nonpositive_object_budget(command: str) -> None:
    result = CliRunner().invoke(boxmot, [command, "--mask-guidance-max-objects", "0"])
    assert result.exit_code == 2
    assert "--mask-guidance-max-objects" in result.output


def test_workflow_shares_matching_edgetam_weights_with_independent_states(monkeypatch, tmp_path: Path) -> None:
    from boxmot.components.artifacts import sha256_artifact
    from boxmot.segmentors import SegmentorSpec
    from boxmot.segmentors.propagation import edgetam

    args = _args(tmp_path, segmentor="edgetam.yaml")
    config = MaskGuidanceConfig(
        args.mask_guidance_weights, device="cpu", max_objects=4,
        min_coverage=0.75, min_fill=0.12, prompt_overlap=0.25,
    )
    spec = SegmentorSpec(
        backend="edgetam",
        artifact=str(config.checkpoint),
        artifact_sha256=sha256_artifact(config.checkpoint),
    )
    shared_model = object()
    created = []

    class Propagator:
        def __init__(self, checkpoint, *, device, max_objects, prompt_overlap):
            assert checkpoint == config.checkpoint and device == "cpu" and max_objects == 4
            assert prompt_overlap == config.prompt_overlap
            self.predictor = shared_model
            self.device = device
            self.max_objects = max_objects
            self.prompt_overlap = prompt_overlap
            created.append(self)

    segmentor = object()

    def create_segmentor(selected, *, model):
        assert selected == spec and model is shared_model
        return segmentor

    monkeypatch.setattr(workflow, "resolve_segmentor_spec", lambda *a, **kw: (spec, {}))
    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", Propagator)
    monkeypatch.setattr(workflow, "create_segmentor", create_segmentor)

    guidance, result = workflow._share_edgetam_model(args, config)

    assert isinstance(guidance, MaskGuidance)
    assert guidance.config is config
    assert guidance._propagator is created[0]
    assert result is segmentor
    assert len(created) == 1


@pytest.mark.parametrize("different", ["weights", "precision", "backend"])
def test_workflow_does_not_share_incompatible_edgetam_components(monkeypatch, tmp_path, different) -> None:
    from boxmot.components.artifacts import sha256_artifact
    from boxmot.segmentors import SegmentorSpec
    from boxmot.segmentors.propagation import edgetam

    args = _args(tmp_path, segmentor="segmentor.yaml")
    config = MaskGuidanceConfig(args.mask_guidance_weights, device="cpu")
    spec = SegmentorSpec(
        backend="sam" if different == "backend" else "edgetam",
        artifact=str(config.checkpoint),
        artifact_sha256="0" * 64 if different == "weights" else sha256_artifact(config.checkpoint),
        precision="fp16" if different == "precision" else "fp32",
    )
    monkeypatch.setattr(workflow, "resolve_segmentor_spec", lambda *a, **kw: (spec, {}))
    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", lambda *a, **kw: pytest.fail("Allocated shared model"))
    assert workflow._share_edgetam_model(args, config) == (config, None)


@pytest.mark.parametrize("show, save", [(True, False), (False, True), (True, True)])
def test_live_outputs_show_guidance_masks_with_identical_save_pixels_and_no_extra_inference(
    monkeypatch, tmp_path, show, save
) -> None:
    """Render only the capped guidance identities, including a frame with no boxes."""
    from boxmot.engine.tracking import sinks as sink_module
    from boxmot.segmentors.propagation import edgetam

    captured = capture_rendered_images(monkeypatch)
    backends = []
    rows = np.array([[10, 20, 45, 80, 0.95, 0], [75, 20, 110, 80, 0.95, 0]], dtype=np.float32)

    class Propagator:
        def __init__(self, checkpoint, *, device, max_objects, prompt_overlap):
            self.device, self.max_objects = device, max_objects
            self.prompt_overlap = prompt_overlap
            self.calls = []
            self.mask_refs = []
            backends.append(self)

        def propagate(self, index, frame, active_boxes, new_boxes):
            if index == 2:
                assert self.mask_refs[0]() is None, "Live rendering retained the previous mask array"
            self.calls.append(index)
            if index == 0:
                return {}
            mask = np.zeros((96, 128), dtype=bool)
            mask[20:80, 10:45] = True
            mask.setflags(write=False)
            self.mask_refs.append(weakref.ref(mask))
            return {0: mask}

        def retain_tracks(self, track_ids):
            pass

        def reset(self):
            pass

    class Source:
        closed = False

        def __iter__(self):
            for index in range(3):
                yield Frame(
                    torch.full((3, 96, 128), 150, dtype=torch.uint8),
                    sample_id=f"live/{index}",
                    sequence_id="live",
                    frame_index=index,
                )

        def close(self):
            self.closed = True

    class Detector:
        capabilities = DetectorCapabilities(provides_masks=True)

        def predict(self, frames):
            detections = []
            for frame in frames:
                values = rows[:0] if frame.frame_index == 1 else rows
                masks = torch.zeros((len(values), 96, 128), dtype=torch.bool)
                for index, box in enumerate(values):
                    x1, y1, x2, y2 = box[:4].astype(int)
                    masks[index, y1:y2, x1:x2] = True
                detections.append(
                    Detections(
                        Boxes(torch.from_numpy(values[:, :4].copy())),
                        scores=torch.from_numpy(values[:, 4].copy()),
                        class_ids=torch.from_numpy(values[:, 5].astype(np.int64)),
                        sample_id=frame.sample_id,
                        masks=MaskBatch(masks),
                    )
                )
            return detections

    real_render = sink_module.render_result
    rendered = []

    def render(frame, result, *, guidance_masks, **kwargs):
        assert guidance_masks is not None
        assert result.tracks.masks is None
        assert result.detections.masks is not None
        if guidance_masks:
            assert guidance_masks[0] is backends[0].mask_refs[-1]()
        rendered.append((len(result.tracks), len(guidance_masks)))
        return real_render(frame, result, guidance_masks=guidance_masks, **kwargs)

    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", Propagator)
    monkeypatch.setattr(sink_module, "render_result", render)
    source = Source()
    args = _args(
        tmp_path,
        show=show,
        save=save,
        save_txt=True,
        project=tmp_path,
        name="rendered",
        mask_guidance_max_objects=1,
        # Preserve detector masks in the result to check they cannot be mistaken
        # for guidance on the initial frame or for identities outside the cap.
        segmentor="provided-by-detector.yaml",
    )
    result = workflow.run_track(args, detector=Detector(), source=source)

    assert source.closed
    assert backends[0].max_objects == 1
    assert backends[0].calls == [0, 1, 2]
    assert rendered == [(2, 0), (0, 1), (2, 1)]
    mot_rows = np.loadtxt(result.mot_path, delimiter=",", ndmin=2)
    np.testing.assert_array_equal(mot_rows[:, :2], [[1, 0], [1, 1], [3, 0], [3, 1]])
    images = captured.shown if show else captured.writers[0].images
    assert len(images) == 3
    np.testing.assert_array_equal(images[0][55, 25], [150, 150, 150])
    assert np.any(images[1][55, 25] != images[0][55, 25])
    np.testing.assert_array_equal(images[1][55, 25], images[2][55, 25])
    for image in images:
        np.testing.assert_array_equal(image[55, 90], [150, 150, 150])
    if save:
        assert captured.writers[0].closed
    if show and save:
        assert len(captured.shown) == len(captured.writers[0].images)
        for preview, encoded in zip(captured.shown, captured.writers[0].images, strict=True):
            np.testing.assert_array_equal(preview, encoded)
