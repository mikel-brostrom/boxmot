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
    "eval",
    "tune",
    "research",
    "train-reid",
    "eval-reid",
    "compare-reid",
    "export",
    "build",
)
EXPECTED_COMMANDS = set(EXPECTED_COMMAND_ORDER)


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


def test_cached_workflows_expose_build_requirements_and_eval_component_selectors() -> None:
    for command in ("eval", "tune", "research"):
        options = _command_options(command)
        help_result = CliRunner().invoke(boxmot, [command, "--help"])
        assert help_result.exit_code == 0
        assert "--build TEXT" in help_result.output
        assert options["build_ref"].required is (command != "eval")
        if command == "eval":
            assert "--detector" in help_result.output
            assert "--reid" in help_result.output
        else:
            assert "--detector" not in help_result.output
            assert "--reid" not in help_result.output
        assert "--imgsz" not in help_result.output
        assert "--conf" not in help_result.output
        assert "--postprocessing" not in help_result.output
        assert "--tracking-backend" not in help_result.output
        if command in {"eval", "tune"}:
            assert "--n-threads" in help_result.output
        else:
            assert "--n-threads" not in help_result.output
    eval_build_help = _command_options("eval")["build_ref"].help
    assert eval_build_help is not None
    assert "--experiment" in eval_build_help
    assert "--detector" in eval_build_help
    assert "materializ" in eval_build_help


@pytest.mark.parametrize(
    ("command", "selector"),
    (
        ("tune", ("--experiment", "mot17/ablation-yolox-lmbn.yaml")),
        ("research", ("--experiment", "mot17/ablation-yolox-lmbn.yaml")),
    ),
)
def test_cached_workflows_reject_missing_explicit_build(command: str, selector: tuple[str, str]) -> None:
    result = CliRunner().invoke(boxmot, [command, *selector])

    assert result.exit_code != 0
    assert "Missing option '--build'" in result.output


def test_eval_dataset_without_build_requires_a_detector() -> None:
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", "mot17"])

    assert result.exit_code == 2
    assert "--detector" in result.output
    assert "--build" in result.output


@pytest.mark.parametrize("option", ("--detector", "--reid"))
def test_eval_experiment_rejects_direct_component_overrides(option: str) -> None:
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture-experiment", option, "component-profile"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--experiment" in result.output
    assert option in result.output


def test_eval_reid_selection_requires_a_detector() -> None:
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--dataset", "mot17", "--reid", "lmbn-n-duke"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--detector" in result.output


def test_eval_direct_components_require_a_dataset() -> None:
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"],
    )

    assert result.exit_code == 2
    assert "No such option" not in result.output
    assert "--dataset" in result.output


def test_eval_direct_components_materialize_and_evaluate_the_matching_authored_experiment(
    monkeypatch,
    tmp_path,
) -> None:
    build_path = tmp_path / "builds" / ("b" * 64)
    calls = []
    captured = {}

    def materialize_main(args):
        calls.append("materialize")
        captured["materialize"] = args
        return build_path

    def eval_main(args):
        calls.append("eval")
        captured["eval"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=eval_main),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["materialize", "eval"]
    expected_experiment = (EXPERIMENT_CONFIGS_DIR / "mot17" / "ablation-yolox-lmbn.yaml").resolve()
    for args in captured.values():
        assert resolve_experiment_path(args.experiment) == expected_experiment
        assert getattr(args, "dataset", None) is None
        assert not hasattr(args, "detector")
        assert not hasattr(args, "reid")
    materialize_args = captured["materialize"]
    assert materialize_args.materialize_split == "ablation"
    assert materialize_args.materialize_mode == "eval"
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is False
    assert materialize_args.publish_embeddings is True
    assert materialize_args.resume is True
    assert "device" not in materialize_args.materialize_explicit_keys
    eval_args = captured["eval"]
    assert eval_args.build == build_path
    assert eval_args.split == "ablation"
    assert eval_args.tracker == "botsort"


def test_eval_without_build_materializes_experiment_then_evaluates(monkeypatch, tmp_path) -> None:
    build_path = tmp_path / "builds" / ("a" * 64)
    calls = []
    captured = {}

    def materialize_main(args):
        calls.append("materialize")
        captured["materialize"] = args
        return build_path

    def eval_main(args):
        calls.append("eval")
        captured["eval"] = args

    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.materialization.workflow",
        SimpleNamespace(main=materialize_main),
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=eval_main),
    )

    data_root = tmp_path / "data"
    build_root = tmp_path / "builds"
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "fixture-experiment",
            "--data-root",
            str(data_root),
            "--build-root",
            str(build_root),
            "--split",
            "ablation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["materialize", "eval"]
    materialize_args = captured["materialize"]
    assert materialize_args.experiment == "fixture-experiment"
    assert materialize_args.data_root == data_root
    assert materialize_args.build_root == build_root
    assert materialize_args.materialize_split == "ablation"
    assert materialize_args.materialize_mode == "eval"
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is False
    assert materialize_args.publish_embeddings is False
    assert materialize_args.resume is True
    assert "device" not in materialize_args.materialize_explicit_keys
    eval_args = captured["eval"]
    assert eval_args.build == build_path
    assert isinstance(eval_args.build, Path)
    assert eval_args.build_root == build_root
    assert eval_args.data_root == data_root
    assert eval_args.experiment == "fixture-experiment"
    assert eval_args.split == "ablation"


@pytest.mark.parametrize(
    "selection",
    (
        ("--experiment", "fixture-experiment"),
        ("--dataset", "mot17", "--detector", "yolox-x-mot17", "--reid", "lmbn-n-duke"),
    ),
)
def test_eval_forwards_explicit_automatic_materialization_device(
    monkeypatch,
    tmp_path,
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
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.setdefault("eval", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        ["eval", *selection, "--device", "mps"],
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
def test_eval_rejects_materialization_device_with_explicit_build(selection: tuple[str, ...]) -> None:
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
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
        ("sam2mot", True, False),
        ("strongsort", False, True),
        ("botsort", False, True),
        ("sfsort", False, False),
    ),
)
def test_eval_automatic_materialization_publishes_tracker_compatible_artifacts(
    monkeypatch,
    tmp_path,
    tracker: str,
    publish_masks: bool,
    publish_embeddings: bool,
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
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: captured.setdefault("eval", args)),
    )

    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture-experiment", "--tracker", tracker],
    )

    assert result.exit_code == 0, result.output
    materialize_args = captured["materialize"]
    assert materialize_args.publish_image_refs is True
    assert materialize_args.publish_masks is publish_masks
    assert materialize_args.publish_embeddings is publish_embeddings


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
def test_eval_does_not_start_after_automatic_materialization_fails(
    monkeypatch,
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
        "boxmot.engine.eval.evaluator",
        SimpleNamespace(main=lambda args: evaluated.append(args)),
    )

    result = CliRunner().invoke(boxmot, ["eval", *selection])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "materialization failed"
    assert evaluated == []


def test_eval_requires_exactly_one_dataset_or_experiment_selector() -> None:
    missing = CliRunner().invoke(boxmot, ["eval", "--build", "build-id"])
    conflicting = CliRunner().invoke(
        boxmot,
        ["eval", "--dataset", "mot17", "--experiment", "experiment", "--build", "build-id"],
    )

    assert missing.exit_code != 0
    assert "requires either --dataset" in missing.output
    assert conflicting.exit_code != 0
    assert "accepts either --dataset" in conflicting.output


@pytest.mark.parametrize("command", ("tune", "research"))
def test_experiment_only_cached_workflows_reject_dataset_or_missing_experiment(command: str) -> None:
    missing = CliRunner().invoke(boxmot, [command, "--build", "build-id"])
    dataset_override = CliRunner().invoke(
        boxmot,
        [command, "--dataset", "mot17", "--build", "build-id"],
    )

    assert missing.exit_code != 0
    assert "requires --experiment" in missing.output
    assert dataset_override.exit_code != 0
    assert "No such option '--dataset'" in dataset_override.output


@pytest.mark.parametrize("direct_components", (False, True))
def test_eval_dispatches_explicit_build(monkeypatch, direct_components: bool) -> None:
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.evaluator",
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

    result = CliRunner().invoke(boxmot, ["eval", *selection])

    assert result.exit_code == 0, result.output
    assert captured["args"].build == "build-0123456789abcdef01234567"
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
