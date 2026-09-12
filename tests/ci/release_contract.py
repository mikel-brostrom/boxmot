"""Shared installed-package release checks, runnable without the source package."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import subprocess

EXPECTED_TRACKERS = (
    ("boosttrack", "BoostTrack"),
    ("botsort", "BotSort"),
    ("bytetrack", "ByteTrack"),
    ("deepocsort", "DeepOcSort"),
    ("eagermot", "EagerMot"),
    ("hybridsort", "HybridSort"),
    ("maf_hda", "MafHda"),
    ("occluboost", "OccluBoost"),
    ("ocsort", "OcSort"),
    ("sfsort", "SFSORT"),
    ("strongsort", "StrongSort"),
)
EXPECTED_PUBLIC_API = ("__version__", "create_tracker", *(public_name for _, public_name in EXPECTED_TRACKERS))
EXPECTED_CLI_COMMANDS = (
    "track",
    "materialize",
    "eval",
    "tune",
    "research",
    "train-reid",
    "eval-reid",
    "compare-reid",
    "export",
    "build",
    "install",
)


def check_release_contract(expected_version: str | None = None) -> None:
    """Check the requested release or installed version and public discovery APIs.

    Explicit release versions also support source-only service images without
    distribution metadata. Otherwise, the installed distribution is the version
    authority; editable installs must be refreshed after source version changes.
    """
    import click

    import boxmot
    from boxmot.engine.cli import boxmot as boxmot_cli
    from boxmot.engine.config.experiments import resolve_experiment_config

    if expected_version is None:
        expected_version = importlib.metadata.version("boxmot")
    assert boxmot.__version__ == expected_version, (
        f"Package version: expected {expected_version!r}, got {boxmot.__version__!r}"
    )
    assert boxmot.__all__ == EXPECTED_PUBLIC_API, (
        f"Public API: expected {EXPECTED_PUBLIC_API!r}, got {boxmot.__all__!r}"
    )
    with click.Context(boxmot_cli) as context:
        commands = tuple(boxmot_cli.list_commands(context))
    assert commands == EXPECTED_CLI_COMMANDS, f"CLI commands: expected {EXPECTED_CLI_COMMANDS!r}, got {commands!r}"
    assert importlib.util.find_spec("boxmot.api") is None
    assert importlib.util.find_spec("boxmot.data") is None
    for name in ("BoxMOT", "Detector", "ReIDModel"):
        assert not hasattr(boxmot, name), f"Removed public alias remains: {name}"
    experiment = resolve_experiment_config("mot17/ablation-yolox-lmbn.yaml")
    assert experiment["detector"]["id"] == "yolox-x-mot17"
    assert experiment["reid"]["id"] == "lmbn-n-duke"


def check_cli_help() -> None:
    """Resolve each advertised command through the installed console entrypoint."""
    for command in EXPECTED_CLI_COMMANDS:
        result = subprocess.run(["boxmot", command, "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"boxmot {command} --help failed:\n{result.stdout}\n{result.stderr}"


def check_tracker_imports() -> dict[str, type]:
    """Resolve every public lazy export, so missing implementation files fail."""
    import boxmot

    classes = {}
    for name, public_name in EXPECTED_TRACKERS:
        tracker_class = getattr(boxmot, public_name)
        assert isinstance(tracker_class, type), f"boxmot.{public_name} is not a tracker class"
        assert tracker_class.__name__ == public_name, f"boxmot.{public_name} resolves to {tracker_class!r}"
        assert callable(getattr(tracker_class, "update", None)), f"boxmot.{public_name} has no update method"
        classes[name] = tracker_class
    return classes


def check_tracker_tracking(name: str, tracker_class: type, geometry: str) -> None:
    """Exercise installed defaults, empty input, and two identities on CPU."""
    import numpy as np
    import torch

    from boxmot import create_tracker
    from boxmot.structures import (
        Boxes,
        Boxes3D,
        CameraModel,
        Detections,
        Detections3D,
        Frame,
        MaskBatch,
        MultimodalTracks,
        OrientedBoxes,
        Tracks,
    )
    from boxmot.trackers import TrackerSpec

    tracker = create_tracker(TrackerSpec(name, geometry=geometry))
    label = f"{name}/{geometry}"
    assert type(tracker) is tracker_class, f"{label}: factory returned {type(tracker)!r}"
    is_obb = geometry == "obb"
    geometry_type = OrientedBoxes if is_obb else Boxes
    values = torch.tensor(
        [[27, 35, 26, 27, 0.2], [75, 47, 26, 39, -0.3]] if is_obb else [[14, 21, 40, 48], [62, 28, 88, 67]],
        dtype=torch.float32,
    )
    height, width = 240, 320
    camera = None
    spatial_values = None
    if tracker.requirements.detections_3d:
        # These image boxes overlap the projected 3D cuboids for sensor fusion.
        values = torch.tensor([[78, 39, 122, 62], [136, 39, 189, 62]], dtype=torch.float32)
        spatial_values = torch.tensor([[0, 1, 10, 0, 4, 2, 2], [6, 1, 10, 0, 4, 2, 2]], dtype=torch.float32)
        camera = CameraModel(
            projection=torch.tensor([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]], dtype=torch.float32),
            image_size=(height, width),
        )
    image = torch.from_numpy(np.random.default_rng(0).integers(0, 256, (3, height, width), dtype=np.uint8))
    masks = torch.zeros((2, height, width), dtype=torch.bool)
    masks[0, 21:48, 14:40] = True
    masks[1, 28:67, 62:88] = True
    embeddings = torch.eye(2, 4, dtype=torch.float32)
    class_ids = torch.tensor([0, 65], dtype=torch.int64)
    sequence_id = "release-smoke"

    def update(frame_index: int, count: int) -> Tracks:
        """Supply required enrichments explicitly and verify canonical output."""
        sample_id = f"{sequence_id}/{frame_index}"
        detections = Detections(
            geometry=geometry_type(values[:count].clone()),
            scores=torch.full((count,), 0.99, dtype=torch.float32),
            class_ids=class_ids[:count].clone(),
            sample_id=sample_id,
            embeddings=embeddings[:count].clone() if tracker.requirements.embeddings else None,
            masks=MaskBatch(masks[:count].clone()) if tracker.requirements.masks else None,
        )
        frame = (
            Frame(image=image.clone(), sample_id=sample_id, sequence_id=sequence_id, frame_index=frame_index)
            if tracker.requirements.frame
            else None
        )
        spatial = (
            Detections3D(
                geometry=Boxes3D(spatial_values[:count].clone()),
                scores=detections.scores.clone(),
                class_ids=detections.class_ids.clone(),
                sample_id=sample_id,
            )
            if spatial_values is not None
            else None
        )
        output = tracker.update(detections, frame, detections_3d=spatial, camera=camera)
        if spatial is not None:
            assert isinstance(output, MultimodalTracks), f"{label}: sensor input did not return MultimodalTracks"
            spatial_tracks = output.spatial_tracks
            assert isinstance(spatial_tracks.geometry, Boxes3D), f"{label}: incorrect spatial geometry"
            assert spatial_tracks.sample_id == sample_id, f"{label}: spatial output belongs to a different frame"
            assert spatial_tracks.geometry.values.shape == (len(spatial_tracks), 7), f"{label}: incorrect 3D layout"
            assert torch.isfinite(spatial_tracks.geometry.values).all(), f"{label}: non-finite spatial output"
            assert torch.equal(spatial_tracks.track_ids, output.image_tracks.track_ids), (
                f"{label}: image and spatial track IDs differ"
            )
            indices = spatial_tracks.detection_indices
            assert sorted(indices.tolist()) == list(range(len(spatial_tracks))), f"{label}: incorrect 3D associations"
            assert torch.equal(spatial_tracks.class_ids, class_ids[indices]), f"{label}: spatial class IDs changed"
            torch.testing.assert_close(spatial_tracks.geometry.values, spatial.geometry.values[indices])
            output = output.image_tracks
        assert isinstance(output, Tracks), f"{label}: canonical input did not return Tracks"
        assert isinstance(output.geometry, geometry_type), f"{label}: incorrect output geometry"
        assert output.sample_id == sample_id, f"{label}: output belongs to a different frame"
        rows = output.to_obb_rows() if is_obb else output.to_aabb_rows()
        assert rows.shape == (len(output), 9 if is_obb else 8), f"{label}: incorrect output layout"
        assert torch.isfinite(rows).all(), f"{label}: non-finite tracking output"
        if tracker.requirements.masks:
            assert output.masks is not None, f"{label}: missing output masks"
            assert output.masks.values.shape == (len(output), height, width), f"{label}: incorrect mask layout"
        return output

    assert len(update(0, 0)) == 0, f"{label}: empty initial input produced tracks"
    previous_ids = None
    for frame_index in range(1, 7):
        output = update(frame_index, 2)
        # Allow the default confirmation period before checking both identities.
        if frame_index >= 5:
            assert len(output) == 2, f"{label}: expected two confirmed tracks, got {len(output)}"
            indices = output.detection_indices
            assert sorted(indices.tolist()) == [0, 1], f"{label}: incorrect detection associations"
            assert torch.equal(output.class_ids, class_ids[indices]), f"{label}: class IDs changed"
            identities = dict(zip(indices.tolist(), output.track_ids.tolist()))
            assert len(set(identities.values())) == 2, f"{label}: duplicate track IDs"
            if previous_ids is not None:
                assert identities == previous_ids, f"{label}: track IDs changed between matching frames"
            previous_ids = identities
    if spatial_values is not None:
        assert len(update(7, 0)) == 0, f"{label}: missing sensor observations emitted stale image tracks"
        recovered = update(8, 2)
        assert dict(zip(recovered.detection_indices.tolist(), recovered.track_ids.tolist())) == previous_ids, (
            f"{label}: track IDs changed after a missing frame"
        )
        tracker.reset()
        sequence_id = "release-smoke-reset"
        assert len(update(0, 0)) == 0, f"{label}: reset retained tracks from the previous sequence"
        for frame_index in range(1, 7):
            restarted = update(frame_index, 2)
        assert dict(zip(restarted.detection_indices.tolist(), restarted.track_ids.tolist())) == previous_ids, (
            f"{label}: reset did not restart independent sequence identities"
        )
    print(f"Tracker smoke passed: {label}", flush=True)


def check_tracker_api() -> None:
    """Import and run all shipped Python trackers without fetching any models."""
    import numpy as np

    from boxmot import ByteTrack

    for name, tracker_class in check_tracker_imports().items():
        for geometry in sorted(kind.value for kind in tracker_class.capabilities.geometry_kinds):
            check_tracker_tracking(name, tracker_class, geometry)

    # Also exercise the direct class constructor and common packed NumPy API.
    tracker = ByteTrack()
    detections = np.array([[14, 21, 40, 48, 0.99, 0]], dtype=np.float32)
    first = tracker.update(detections)
    second = tracker.update(detections)
    for output in (first, second):
        assert isinstance(output, np.ndarray) and output.shape == (1, 8), "ByteTrack: invalid NumPy output"
        assert output.dtype == np.float64 and output.flags.c_contiguous, "ByteTrack: invalid packed layout"
        assert np.isfinite(output).all(), "ByteTrack: non-finite NumPy output"
    np.testing.assert_array_equal(first[:, 4], second[:, 4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="Expected runtime version; defaults to the installed BoxMOT distribution version.",
    )
    parser.add_argument("--check-cli-help", action="store_true", help="Also exercise every installed command's help.")
    parser.add_argument(
        "--check-trackers", action="store_true", help="Import every public tracker and exercise CPU tracking."
    )
    args = parser.parse_args()
    check_release_contract(expected_version=args.expected_version)
    if args.check_cli_help:
        check_cli_help()
    if args.check_trackers:
        check_tracker_api()
    print("Requested release contract checks passed.")
