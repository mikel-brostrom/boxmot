from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.experiment_config import EXPERIMENT_CONFIGS_DIR, resolve_experiment_path

EXPECTED_COMMAND_ORDER = (
    "track",
    "materialize",
    "time-variant",
    "eval",
    "eval-trackrcnn",
    "eval-eagermot",
    "tune",
    "tune-eagermot",
    "research",
    "train-reid",
    "eval-reid",
    "compare-reid",
    "export",
    "build",
)
EXPECTED_COMMANDS = set(EXPECTED_COMMAND_ORDER)
CACHED_WORKFLOW_MODULES = {
    "eval": "boxmot.engine.eval.evaluator",
    "tune": "boxmot.engine.tuning.tuner",
}


def _command_options(command_name: str) -> dict[str, click.Option]:
    command = boxmot.get_command(click.Context(boxmot), command_name)
    assert command is not None
    return {param.name: param for param in command.params if isinstance(param, click.Option)}


def test_v24_command_set_is_exact() -> None:
    command_names = boxmot.list_commands(click.Context(boxmot))
    assert command_names == list(EXPECTED_COMMAND_ORDER)
    assert set(command_names) == EXPECTED_COMMANDS
    assert "generate" not in command_names

    removed = CliRunner().invoke(boxmot, ["generate", "--help"])
    assert removed.exit_code != 0
    assert "No such command 'generate'" in removed.output


def test_reid_evaluation_path_options_preserve_their_click_contracts() -> None:
    for command_name in ("eval-reid", "compare-reid"):
        options = _command_options(command_name)
        weights = options["weights"].type
        output = options["output"].type

        assert isinstance(weights, click.Path)
        assert weights.exists is True
        assert weights.file_okay is True
        assert weights.dir_okay is True
        assert weights.type is None
        assert isinstance(output, click.Path)
        assert output.file_okay is True
        assert output.dir_okay is True
        assert output.type is None

    data_dir = _command_options("eval-reid")["data_dir"].type
    assert isinstance(data_dir, click.Path)
    assert data_dir.exists is True
    assert data_dir.file_okay is True
    assert data_dir.dir_okay is True
    assert data_dir.type is None


def test_train_reid_click_defaults_match_the_promoted_default_recipe() -> None:
    options = _command_options("train-reid")
    expected = {
        "epochs": 200,
        "gradual_unfreeze_head_epochs": 5,
        "gradual_unfreeze_stage_epochs": 20,
        "gradual_unfreeze_backbone_lr_mult": 0.1,
        "gradual_unfreeze_backbone_lr_epochs": 5,
        "p_ids": 12,
        "k_instances": 8,
        "triplet_soft_margin": True,
        "scale_balanced_branches": True,
        "feature_fusion": "global_final_parts_stage0_semantic_fine",
        "spatial_conv_mode": "depthwise_separable",
        "attention_window_layout": "rect",
        "interpolate_pretrained_attention_bias": True,
        "attention_mask": True,
        "head_parts": "1,2,4",
        "device": "mps",
    }

    assert {name: options[name].default for name in expected} == expected


def test_train_reid_implicit_click_defaults_do_not_override_an_explicit_model(
    monkeypatch,
    tmp_path,
) -> None:
    from boxmot.engine.commands.reid import train as train_command
    from boxmot.reid.datasets import resolution

    captured = {}
    monkeypatch.setattr(resolution, "resolve_reid_train_data", lambda args: args)
    monkeypatch.setattr(train_command, "main", lambda args: captured.setdefault("args", args))

    result = CliRunner().invoke(
        boxmot,
        ["train-reid", "--model", "csl_tinyvit_7m", "--data-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.epochs == 250
    assert args.p_ids == 16
    assert args.k_instances == 4
    assert args.feature_fusion == "last2"
    assert args.head_parts == (1, 2)
    assert args.device == "cpu"


def test_materialize_requires_an_experiment() -> None:
    options = _command_options("materialize")

    assert options["experiment"].required is True

    missing = CliRunner().invoke(boxmot, ["materialize"])
    assert missing.exit_code == 2
    assert "Missing option '--experiment'" in missing.output


@pytest.mark.parametrize("command", ("materialize", "eval", "tune", "research"))
def test_experiment_option_documents_yaml_filename_selector(command: str) -> None:
    help_text = _command_options(command)["experiment"].help

    assert help_text is not None
    assert "experiment YAML filename or path" in help_text
    assert "experiment id" not in help_text.casefold()


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--dataset", "mot17"),
        ("--source", "frames"),
        ("--split", "test"),
        ("--detector", "detector.yaml"),
        ("--segmentor", "segmentor.yaml"),
        ("--reid", "reid.yaml"),
        ("--geometry", "obb"),
    ),
)
def test_materialize_rejects_non_experiment_semantic_selectors(option: str, value: str) -> None:
    result = CliRunner().invoke(
        boxmot,
        ["materialize", "--experiment", "exp", option, value],
    )

    assert result.exit_code == 2
    assert f"No such option '{option}'" in result.output


def test_materialize_option_and_dispatch_contract_is_experiment_only(monkeypatch) -> None:
    options = _command_options("materialize")
    assert set(options) == {
        "build_root",
        "data_root",
        "device",
        "experiment",
        "fps",
        "plan_overrides",
        "plan_path",
        "publish_embeddings",
        "publish_image_refs",
        "publish_masks",
        "resume",
    }

    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(boxmot, ["materialize", "--experiment", "exp"])

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.experiment == "exp"
    assert set(vars(args)) == {
        *options,
        "materialize_explicit_keys",
    }


def test_experiment_materialization_accepts_explicit_execution_device(monkeypatch) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        ["materialize", "--experiment", "exp", "--device", "mps"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].device == "mps"
    assert "device" in captured["args"].materialize_explicit_keys


def test_materialization_default_device_is_not_an_override(monkeypatch) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(boxmot, ["materialize", "--experiment", "exp"])

    assert result.exit_code == 0, result.output
    assert captured["args"].device == "cpu"
    assert "device" not in captured["args"].materialize_explicit_keys


def test_cached_workflows_expose_build_requirements_and_component_selectors() -> None:
    for command in ("eval", "tune", "research"):
        options = _command_options(command)
        help_result = CliRunner().invoke(boxmot, [command, "--help"])
        assert help_result.exit_code == 0
        assert "--build TEXT" in help_result.output
        assert options["build_ref"].required is (command == "research")
        if command in {"eval", "tune"}:
            assert "--detector" in help_result.output
            assert "--reid" in help_result.output
            assert "--device" in help_result.output
        else:
            assert "--detector" not in help_result.output
            assert "--reid" not in help_result.output
        assert "--imgsz" not in help_result.output
        assert "--conf" not in help_result.output
        assert "--postprocessing" not in help_result.output
        assert "--tracking-backend" not in help_result.output
        assert "--n-threads" not in help_result.output
        if command in {"eval", "tune"}:
            assert "--sequence-workers" in help_result.output
        else:
            assert "--sequence-workers" not in help_result.output
    for command in ("eval", "tune"):
        build_help = _command_options(command)["build_ref"].help
        assert build_help is not None
        assert "--experiment" in build_help
        assert "--detector" in build_help
        assert "materializ" in build_help


@pytest.mark.parametrize("command", ("eval", "tune"))
def test_sequence_workers_reaches_cached_workflow_namespace(monkeypatch, command: str) -> None:
    captured = {}
    module = "boxmot.engine.eval.evaluator" if command == "eval" else "boxmot.engine.tuning.tuner"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=lambda args: captured.setdefault("args", args)))

    result = CliRunner().invoke(
        boxmot,
        [command, "--experiment", "fixture-experiment", "--build", "fixture-build", "--sequence-workers", "3"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].sequence_workers == 3
    assert not hasattr(captured["args"], "n_threads")


@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize("eval_masks", (False, True))
def test_mask_evaluation_selection_reaches_cached_workflow(monkeypatch, command: str, eval_masks: bool) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            command,
            "--dataset",
            "kitti-mots",
            "--build",
            "fixture-build",
            *(["--eval-masks"] if eval_masks else []),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].eval_masks is eval_masks
    assert _command_options(command)["eval_masks"].default is False


@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize(
    "option, value, error",
    [
        ("--n-threads", "2", "No such option '--n-threads'"),
        ("--sequence-workers", "0", "Invalid value for '--sequence-workers'"),
        ("--sequence-workers", "-1", "Invalid value for '--sequence-workers'"),
        ("--sequence-workers", "1.5", "Invalid value for '--sequence-workers'"),
    ],
)
def test_sequence_worker_options_fail_before_workflow_dispatch(monkeypatch, command, option, value, error) -> None:
    calls = []
    module = "boxmot.engine.eval.evaluator" if command == "eval" else "boxmot.engine.tuning.tuner"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(main=lambda args: calls.append(args)))

    result = CliRunner().invoke(
        boxmot,
        [command, "--experiment", "fixture-experiment", "--build", "fixture-build", option, value],
    )

    assert result.exit_code == 2
    assert error in result.output
    assert calls == []


def test_research_rejects_missing_explicit_build() -> None:
    result = CliRunner().invoke(boxmot, ["research", "--experiment", "mot17/ablation-yolox-lmbn.yaml"])

    assert result.exit_code != 0
    assert "Missing option '--build'" in result.output


@pytest.mark.parametrize("command", ("eval", "tune"))
def test_dataset_without_build_requires_a_detector(command: str) -> None:
    result = CliRunner().invoke(boxmot, [command, "--dataset", "mot17"])

    assert result.exit_code == 2
    assert "--detector" in result.output
    assert "--build" in result.output


@pytest.mark.parametrize("option", ("--detector", "--reid"))
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_experiment_rejects_direct_component_overrides(command: str, option: str) -> None:
    result = CliRunner().invoke(
        boxmot,
        [command, "--experiment", "fixture-experiment", option, "component-profile"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--experiment" in result.output
    assert option in result.output


@pytest.mark.parametrize("command", ("eval", "tune"))
def test_reid_selection_requires_a_detector(command: str) -> None:
    result = CliRunner().invoke(
        boxmot,
        [command, "--dataset", "mot17", "--reid", "lmbn-n-duke"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--detector" in result.output


@pytest.mark.parametrize("command", ("eval", "tune"))
def test_direct_components_require_a_dataset(command: str) -> None:
    result = CliRunner().invoke(
        boxmot,
        [command, "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--dataset" in result.output


@pytest.mark.parametrize("fps", (None, 5.0, 2.5))
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_direct_components_materialize_before_replaying_the_matching_authored_experiment(
    monkeypatch,
    tmp_path,
    command: str,
    fps: float | None,
) -> None:
    from boxmot.engine import experiment_config

    build_path = tmp_path / "builds" / ("b" * 64)
    calls = []
    captured = {}
    resolution_modes = []
    resolve = experiment_config.resolve_matching_experiment_path

    def resolve_selection(**kwargs):
        resolution_modes.append(kwargs["mode"])
        return resolve(**kwargs)

    monkeypatch.setattr(experiment_config, "resolve_matching_experiment_path", resolve_selection)

    def materialize_main(args):
        calls.append("materialize")
        captured["materialize"] = args
        return build_path

    def replay_main(args):
        calls.append(command)
        captured[command] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=replay_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            command,
            "--dataset",
            "mot17",
            "--split",
            "ablation",
            "--detector",
            "yolox-x-mot17",
            "--reid",
            "lmbn-n-duke",
            "--tracker",
            "botsort",
            *([] if fps is None else ["--fps", str(fps)]),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["materialize", command]
    assert resolution_modes == [command]
    expected_experiment = (EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml").resolve()
    for args in captured.values():
        assert resolve_experiment_path(args.experiment) == expected_experiment
        assert getattr(args, "dataset", None) is None
        assert not hasattr(args, "detector")
        assert not hasattr(args, "reid")
        assert args.fps == fps
    materialize_args = captured["materialize"]
    assert materialize_args.materialize_split == "ablation"
    assert materialize_args.materialize_mode == command
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is False
    assert materialize_args.publish_embeddings is True
    assert materialize_args.resume is True
    assert "device" not in materialize_args.materialize_explicit_keys
    assert ("fps" in materialize_args.materialize_explicit_keys) is (fps is not None)
    replay_args = captured[command]
    assert replay_args.build == build_path
    assert replay_args.split == "ablation"
    assert replay_args.tracker == "botsort"


@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize("calibrate_kf", (False, True))
def test_without_build_materializes_experiment_once_before_replay(
    monkeypatch, tmp_path, command: str, calibrate_kf: bool
) -> None:
    build_path = tmp_path / "builds" / ("a" * 64)
    calls = []
    captured = {}

    def materialize_main(args):
        calls.append("materialize")
        captured["materialize"] = args
        return build_path

    def replay_main(args):
        calls.append(command)
        captured[command] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=replay_main),
    )

    data_root = tmp_path / "data"
    build_root = tmp_path / "builds"
    result = CliRunner().invoke(
        boxmot,
        [
            command,
            "--experiment",
            "fixture-experiment",
            "--data-root",
            str(data_root),
            "--build-root",
            str(build_root),
            "--split",
            "ablation",
            *(["--n-trials", "200"] if command == "tune" else []),
            *(["--calibrate-kf"] if calibrate_kf else []),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["materialize", command]
    materialize_args = captured["materialize"]
    assert materialize_args.experiment == "fixture-experiment"
    assert materialize_args.data_root == data_root
    assert materialize_args.build_root == build_root
    assert materialize_args.materialize_split == "ablation"
    assert materialize_args.materialize_mode == command
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is False
    assert materialize_args.publish_embeddings is False
    assert materialize_args.resume is True
    assert "device" not in materialize_args.materialize_explicit_keys
    replay_args = captured[command]
    assert replay_args.build == build_path
    assert isinstance(replay_args.build, Path)
    assert replay_args.build_root == build_root
    assert replay_args.data_root == data_root
    assert replay_args.experiment == "fixture-experiment"
    assert replay_args.split == "ablation"
    assert replay_args.calibrate_kf is calibrate_kf
    if command == "tune":
        assert replay_args.n_trials == 200


@pytest.mark.parametrize(
    "selection",
    (
        ("--experiment", "fixture-experiment"),
        ("--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"),
    ),
)
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_forwards_explicit_automatic_materialization_device(
    monkeypatch,
    tmp_path,
    command: str,
    selection: tuple[str, ...],
) -> None:
    captured = {}
    build_path = tmp_path / "build"

    def materialize_main(args):
        captured["materialize"] = args
        return build_path

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=lambda args: captured.setdefault(command, args)),
    )

    result = CliRunner().invoke(
        boxmot,
        [command, *selection, "--device", "mps"],
    )

    assert result.exit_code == 0, result.output
    assert captured["materialize"].device == "mps"
    assert "device" in captured["materialize"].materialize_explicit_keys


@pytest.mark.parametrize(
    "selection",
    (
        ("--experiment", "fixture-experiment"),
        ("--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"),
    ),
)
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_rejects_materialization_device_with_explicit_build(command: str, selection: tuple[str, ...]) -> None:
    result = CliRunner().invoke(
        boxmot,
        [
            command,
            *selection,
            "--build",
            "a" * 64,
            "--device",
            "mps",
        ],
    )

    assert result.exit_code == 2
    assert "--device applies only when --build is omitted" in result.output


@pytest.mark.parametrize(
    ("tracker", "publish_masks", "publish_embeddings"),
    (
        ("maf_hda", True, False),
        ("strongsort", False, True),
        ("botsort", False, True),
        ("sfsort", False, False),
    ),
)
@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize("eval_masks", (False, True))
def test_automatic_materialization_publishes_tracker_compatible_artifacts(
    monkeypatch,
    tmp_path,
    command: str,
    tracker: str,
    publish_masks: bool,
    publish_embeddings: bool,
    eval_masks: bool,
) -> None:
    captured = {}
    build_path = tmp_path / "build"
    experiment_path = tmp_path / "kitti-experiment.yaml"
    experiment_path.write_text(
        "dataset:\n  ref: kitti-mots\n  split: train\n"
        "detector:\n  ref: yolo26n\n  checkpoint: default\n"
        "evaluation:\n  class_map:\n    car: car\n    pedestrian: person\n"
    )

    def materialize_main(args):
        captured["materialize"] = args
        return build_path

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=lambda args: captured.setdefault(command, args)),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            command,
            "--experiment",
            str(experiment_path),
            "--tracker",
            tracker,
            *(["--eval-masks"] if eval_masks else []),
        ],
    )

    assert result.exit_code == 0, result.output
    materialize_args = captured["materialize"]
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is (publish_masks or eval_masks)
    assert materialize_args.publish_embeddings is publish_embeddings
    assert captured[command].eval_masks is eval_masks


@pytest.mark.parametrize("command", ("eval", "tune"))
@pytest.mark.parametrize(
    "selection",
    (
        ("--experiment", "mot17/ablation-yolox-lmbn.yaml"),
        ("--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"),
    ),
)
def test_mask_evaluation_rejects_other_datasets_before_materialization(
    monkeypatch, command: str, selection: tuple[str, ...]
) -> None:
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda args: calls.append(args)),
    )

    result = CliRunner().invoke(boxmot, [command, *selection, "--eval-masks"])

    assert result.exit_code == 2
    assert "--eval-masks requires a KITTI-MOTS dataset" in result.output
    assert calls == []


def test_eval_noncanonical_opt_in_requires_explicit_build() -> None:
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture-experiment", "--allow-noncanonical-build"],
    )

    assert result.exit_code == 2
    assert "--allow-noncanonical-build requires an explicit --build" in result.output


@pytest.mark.parametrize(
    "selection",
    (
        ("--experiment", "fixture-experiment"),
        ("--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"),
    ),
)
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_replay_does_not_start_after_automatic_materialization_fails(
    monkeypatch,
    command: str,
    selection: tuple[str, ...],
) -> None:
    evaluated = []

    def fail_materialization(_args):
        raise RuntimeError("materialization failed")

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=fail_materialization),
    )
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=lambda args: evaluated.append(args)),
    )

    result = CliRunner().invoke(boxmot, [command, *selection])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "materialization failed"
    assert evaluated == []


@pytest.mark.parametrize("command", ("eval", "tune"))
def test_replay_requires_exactly_one_dataset_or_experiment_selector(command: str) -> None:
    missing = CliRunner().invoke(boxmot, [command, "--build", "build-id"])
    conflicting = CliRunner().invoke(
        boxmot,
        [command, "--dataset", "mot17", "--experiment", "experiment", "--build", "build-id"],
    )

    assert missing.exit_code != 0
    assert "requires either --dataset" in missing.output
    assert conflicting.exit_code != 0
    assert "accepts either --dataset" in conflicting.output


def test_research_rejects_dataset_or_missing_experiment() -> None:
    missing = CliRunner().invoke(boxmot, ["research", "--build", "build-id"])
    dataset_override = CliRunner().invoke(
        boxmot,
        ["research", "--dataset", "mot17", "--build", "build-id"],
    )

    assert missing.exit_code != 0
    assert "requires --experiment" in missing.output
    assert dataset_override.exit_code != 0
    assert "No such option '--dataset'" in dataset_override.output


@pytest.mark.parametrize("direct_components", (False, True))
@pytest.mark.parametrize("command", ("eval", "tune"))
def test_replay_dispatches_explicit_build_without_materialization(
    monkeypatch, command: str, direct_components: bool
) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        CACHED_WORKFLOW_MODULES[command],
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=lambda _args: pytest.fail("explicit builds must bypass materialization")),
    )

    selection = ["--dataset", "mot17", "--build", "build-0123456789abcdef01234567"]
    if direct_components:
        selection.extend(("--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"))

    result = CliRunner().invoke(boxmot, [command, *selection])

    assert result.exit_code == 0, result.output
    assert captured["args"].build == "build-0123456789abcdef01234567"
    if command == "eval":
        assert captured["args"].allow_noncanonical_build is False
    if direct_components:
        assert (
            resolve_experiment_path(captured["args"].experiment)
            == (EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml").resolve()
        )
        assert captured["args"].dataset is None
        assert not hasattr(captured["args"], "detector")
        assert not hasattr(captured["args"], "reid")
    else:
        assert captured["args"].dataset == "mot17"


def test_eval_noncanonical_build_requires_explicit_opt_in_and_dispatches(monkeypatch) -> None:
    option = _command_options("eval")["allow_noncanonical_build"]
    assert option.is_flag is True
    assert option.default is False

    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            "mmot",
            "--build",
            "faf16842d9d15fc048a50247df8ae927e7299d5760c9f461af4581cabfa279e6",
            "--allow-noncanonical-build",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].allow_noncanonical_build is True


def test_eval_sequence_option_is_repeatable_and_dispatches_in_order(monkeypatch) -> None:
    options = _command_options("eval")
    sequence_option = options["sequence_names"]
    assert sequence_option.multiple is True
    assert sequence_option.metavar == "NAME"

    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.setdefault("args", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            "mmot",
            "--build",
            "build-0123456789abcdef01234567",
            "--sequence",
            "data23-1",
            "--sequence",
            "data23-2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].sequence_names == ("data23-1", "data23-2")


@pytest.mark.parametrize("tracker", (None, "bytetrack"))
def test_trackrcnn_evaluation_dispatches_saved_paths_and_class_separated_python_tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tracker: str | None
) -> None:
    """Saved detector input bypasses perception setup and retains tracker overrides."""
    captured = {}

    def run(args: SimpleNamespace) -> Path:
        captured["args"] = args
        return tmp_path / "results"

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.trackrcnn", SimpleNamespace(run_trackrcnn=run))
    config = tmp_path / "tracker.yaml"
    config.write_text("det_thresh: 0.75\n", encoding="utf-8")
    arguments = [
        "eval-trackrcnn",
        "--detections",
        str(tmp_path),
        "--images",
        str(tmp_path),
        "--instances",
        str(tmp_path),
        "--tracker-config",
        str(config),
        "--sequence",
        "0006",
        "--sequence",
        "0002",
    ]
    if tracker is not None:
        arguments.extend(("--tracker", tracker))
    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.tracker == (tracker or "maf_hda")
    assert args.tracker_backend == "python"
    assert args.per_class is True
    assert args.detections == args.images == args.instances == tmp_path
    assert args.tracker_config == str(config)
    assert args.sequence_names == ("0006", "0002")
    assert args.split == "val"
    assert args.project == Path("runs/trackrcnn")
    assert f"Results: {tmp_path / 'results'}" in result.output


@pytest.mark.parametrize("error_type", (ValueError, FileNotFoundError, ImportError))
def test_trackrcnn_evaluation_reports_actionable_runner_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    """Data and capability failures are concise CLI errors instead of tracebacks."""
    message = "Selected tracker requires unavailable sensor inputs."

    def fail(_args: SimpleNamespace) -> Path:
        raise error_type(message)

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.trackrcnn", SimpleNamespace(run_trackrcnn=fail))
    result = CliRunner().invoke(
        boxmot,
        ["eval-trackrcnn", "--detections", str(tmp_path), "--images", str(tmp_path), "--instances", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert "Traceback" not in result.output
