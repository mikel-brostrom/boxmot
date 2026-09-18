"""Validate EdgeTAM replay selection before detector materialization or execution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.trackers import edgetam_checkpoint
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture

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


@pytest.mark.parametrize("command", ["track", "eval", "tune"])
@pytest.mark.parametrize(
    "flags,enabled,weights",
    [
        ([], False, None),
        (["--mask-guidance-weights", "custom.pt"], False, Path("custom.pt")),
        (["--edgetam"], True, None),
        (["--edgetam", "--mask-guidance-weights", "custom.pt"], True, Path("custom.pt")),
        (["--edgetam", "--no-edgetam", "--mask-guidance-weights", "custom.pt"], False, Path("custom.pt")),
        (["--no-edgetam", "--edgetam"], True, None),
    ],
)
def test_edgetam_toggle_is_independent_of_checkpoint_selection(
    monkeypatch, command: str, flags: list[str], enabled: bool, weights: Path | None
) -> None:
    received = []
    modules = {
        "track": "boxmot.engine.tracking.workflow",
        "eval": "boxmot.engine.eval.evaluator",
        "tune": "boxmot.engine.tuning.tuner",
    }
    monkeypatch.setitem(sys.modules, modules[command], SimpleNamespace(main=received.append))
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object() if enabled else None)
    evaluation = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(evaluation, "find_spec", lambda _name: object() if enabled else None)
    inputs = (
        ["--source", "clip.mp4"]
        if command == "track"
        else ["--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "existing-build"]
    )
    result = CliRunner().invoke(boxmot, [command, *inputs, *flags])
    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0].edgetam is enabled
    assert received[0].mask_guidance_weights == weights
    assert edgetam_checkpoint(received[0]) == ((weights or Path("edgetam.pt")) if enabled else None)


@pytest.mark.parametrize("command", ["eval", "tune"])
def test_disabled_edgetam_does_not_restrict_native_replay(monkeypatch, command: str) -> None:
    received = []
    module = "boxmot.engine.eval.evaluator" if command == "eval" else "boxmot.engine.tuning.tuner"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=received.append))
    result = CliRunner().invoke(
        boxmot,
        [
            command,
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "existing-build",
            "--tracker-backend",
            "cpp",
            "--no-edgetam",
            "--mask-guidance-weights",
            "missing-checkpoint.pt",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0].tracker_backend == "cpp"
    assert received[0].mask_guidance_weights == Path("missing-checkpoint.pt")
    assert edgetam_checkpoint(received[0]) is None


@pytest.fixture(autouse=True)
def _edge_tam_available(monkeypatch):
    """CLI validation must not require an optional model installation."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(command, "find_spec", lambda _name: object())


def _flags(tmp_path: Path) -> list[str]:
    """Use the full authored MOT17 ablation with a local checkpoint placeholder."""
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    return [
        "eval",
        "--experiment",
        "mot17/ablation-yolox-lmbn.yaml",
        "--tracker",
        "bytetrack",
        "--edgetam",
        "--mask-guidance-weights",
        str(checkpoint),
    ]


@pytest.mark.parametrize("workers", (None, 2))
@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
def test_guidance_allows_replay_device_with_existing_build(monkeypatch, tmp_path, workers, tracker_name) -> None:
    received = []
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=received.append),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda _args: pytest.fail("Existing build requested materialization")),
    )
    flags = [
        *_flags(tmp_path),
        "--tracker",
        tracker_name,
        "--asso-func",
        "iou",
        "--build",
        "existing-build",
        "--device",
        "mps",
    ]
    if workers is not None:
        flags += ["--sequence-workers", str(workers)]

    result = CliRunner().invoke(boxmot, flags)

    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0].build == "existing-build"
    assert received[0].tracker == tracker_name
    assert received[0].device == "mps"
    assert received[0].mask_guidance_weights == tmp_path / "edgetam.pt"
    assert received[0].sequence_workers == (1 if workers is None else workers)


@pytest.mark.parametrize("checkpoint", ("edgetam.pt", "models/edgetam.pt"))
def test_guidance_eval_cli_accepts_checkpoint_to_download(monkeypatch, tmp_path, checkpoint) -> None:
    """Downloadable checkpoint selection survives parsing for cached evaluation."""
    monkeypatch.chdir(tmp_path)
    received = []
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=received.append),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda _args: pytest.fail("Existing build requested materialization")),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--tracker",
            "bytetrack",
            "--build",
            "existing-build",
            "--edgetam",
            "--mask-guidance-weights",
            checkpoint,
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert received[0].mask_guidance_weights == Path(checkpoint)
    assert not Path(checkpoint).exists()


@pytest.mark.parametrize("component_selection", (False, True))
@pytest.mark.parametrize("fps", (None, 15))
def test_guidance_materializes_boxes_and_preserves_checkpoint_for_replay(
    monkeypatch, tmp_path, component_selection, fps
) -> None:
    materialized = []
    evaluated = []

    def materialize(args):
        materialized.append(args)
        return tmp_path / "build"

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=evaluated.append),
    )

    flags = _flags(tmp_path)
    if component_selection:
        flags[1:3] = [
            "--dataset",
            "mot17",
            "--split",
            "ablation",
            "--detector",
            "yolox-x-mot17",
            "--reid",
            "lmbn-n-duke",
        ]
    if fps is not None:
        flags += ["--fps", str(fps)]
    result = CliRunner().invoke(boxmot, [*flags, "--device", "mps"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert len(materialized) == len(evaluated) == 1
    assert materialized[0].device == evaluated[0].device == "mps"
    assert materialized[0].fps == evaluated[0].fps == fps
    assert materialized[0].publish_image_refs is True
    assert materialized[0].publish_masks is False
    assert materialized[0].publish_embeddings is False
    assert not hasattr(materialized[0], "mask_guidance_weights")
    assert evaluated[0].mask_guidance_weights == tmp_path / "edgetam.pt"
    assert evaluated[0].build == tmp_path / "build"
    assert evaluated[0].sequence_names == ()


@pytest.mark.parametrize(
    "extra_flags, message",
    [
        (["--tracker", "maf_hda"], "Python 2D box tracker"),
        (["--tracker", "eagermot"], "requires 3D detections"),
        (["--tracker", "hybridsort"], "asso_func='iou'"),
        (["--tracker-backend", "cpp"], "Python 2D box tracker"),
        (["--per-class"], "does not support --per-class"),
        (["--asso-func", "giou"], "asso_func='iou'"),
        (["--eval-masks"], "box metrics from image datasets only"),
        (["--eval-3d"], "box metrics from image datasets only"),
    ],
)
def test_incompatible_guidance_flags_fail_before_materialization(monkeypatch, tmp_path, extra_flags, message) -> None:
    for module in ("boxmot.engine.materialization.workflow", "boxmot.engine.eval.evaluator"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=lambda _args: pytest.fail("Unexpected workflow")))

    result = CliRunner().invoke(boxmot, [*_flags(tmp_path), *extra_flags])

    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output


@pytest.mark.parametrize(
    "settings, message", (("per_class: true", "per_class=False"), ("asso_func: giou", "asso_func='iou'"))
)
def test_guidance_validates_effective_tracker_configuration(tmp_path, settings, message) -> None:
    config = tmp_path / "tracker.yaml"
    config.write_text(settings + "\n")

    result = CliRunner().invoke(boxmot, [*_flags(tmp_path), "--tracker-config", str(config)])

    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output


@pytest.mark.parametrize("saved_boxes", (False, True))
def test_guidance_rejects_sensor_and_saved_predictions_before_input_loading(monkeypatch, tmp_path, saved_boxes) -> None:
    fixture = sensor_dataset_fixture(tmp_path / "dataset")
    if saved_boxes:
        config = yaml.safe_load(fixture.dataset.read_text())
        for role in ("calibration", "poses", "detections_3d"):
            del config["modalities"][role]
        config["modalities"]["detections_2d"]["options"] = {"load_masks": False}
        fixture.dataset.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.touch()
    for module in ("boxmot.engine.materialization.workflow", "boxmot.engine.eval.evaluator"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=lambda _args: pytest.fail("Unexpected workflow")))

    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            str(fixture.dataset),
            "--tracker",
            "bytetrack",
            "--edgetam",
            "--mask-guidance-weights",
            str(checkpoint),
        ],
    )

    assert result.exit_code == 2, (result.output, result.exception)
    assert "does not support saved-detection or sensor datasets" in result.output


def test_guidance_rejects_oriented_experiment_before_materialization(monkeypatch, tmp_path) -> None:
    from boxmot.engine.config import experiments

    monkeypatch.setattr(
        experiments, "resolve_experiment_config", lambda *_args, **_kwargs: {"dataset": {"box_type": "obb"}}
    )

    result = CliRunner().invoke(boxmot, _flags(tmp_path))

    assert result.exit_code == 2, (result.output, result.exception)
    assert "requires an AABB dataset" in result.output


def test_missing_edge_tam_fails_before_materialization(monkeypatch, tmp_path) -> None:
    command = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(command, "find_spec", lambda _name: None)
    monkeypatch.setattr(command, "_prepare_replay_build", lambda *_args, **_kwargs: pytest.fail("Materialized data"))

    result = CliRunner().invoke(boxmot, _flags(tmp_path))

    assert result.exit_code == 2, (result.output, result.exception)
    assert "EdgeTAM is not installed" in result.output
    assert "--group mask-guidance" in result.output
