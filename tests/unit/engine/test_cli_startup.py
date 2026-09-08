"""Import contracts that keep CLI startup independent of ML runtimes."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests._paths import REPO_ROOT

_HEAVY_RUNTIME_MODULES = (
    "cv2",
    "numpy",
    "torch",
    "boxmot.reid.training.evaluator",
    "boxmot.reid.training.losses",
)

_MODEL_RUNTIME_MODULES = (
    "ultralytics",
    "boxmot.detectors.backends.ultralytics",
    "boxmot.segmentors.backends.sam",
    "boxmot.segmentors.backends.maskrcnn",
    "boxmot.native.trackers",
    "boxmot.native.reid",
    "boxmot.reid.backends.native",
    "boxmot.reid.core.runtime",
    "boxmot.reid.training.evaluator",
    "boxmot.reid.training.losses",
)

_CONFIG_RUNTIME_MODULES = (*_HEAVY_RUNTIME_MODULES, *_MODEL_RUNTIME_MODULES, "pyarrow")

_ROOT_HELP_RUNTIME_MODULES = (
    "cv2",
    "torch",
    "ultralytics",
    "boxmot.detectors",
    "boxmot.trackers",
    "boxmot.reid",
)

_REGISTERED_COMMAND_MODULES = (
    "boxmot.engine.commands.build",
    "boxmot.engine.commands.eval",
    "boxmot.engine.commands.materialize",
    "boxmot.engine.commands.research",
    "boxmot.engine.commands.track",
    "boxmot.engine.commands.tune",
    "boxmot.engine.commands.reid.compare",
    "boxmot.engine.commands.reid.evaluate",
    "boxmot.engine.commands.reid.export",
    "boxmot.engine.commands.reid.train",
)

_REID_COMMAND_MODULES = (
    "boxmot.engine.commands.reid.ablation",
    "boxmot.engine.commands.reid.compare",
    "boxmot.engine.commands.reid.evaluate",
    "boxmot.engine.commands.reid.export",
    "boxmot.engine.commands.reid.human_pretrain",
    "boxmot.engine.commands.reid.privileged_cache",
    "boxmot.engine.commands.reid.teacher_extract",
    "boxmot.engine.commands.reid.train",
)

_COMMAND_MODULE_BY_NAME = {
    "track": "boxmot.engine.commands.track",
    "materialize": "boxmot.engine.commands.materialize",
    "eval": "boxmot.engine.commands.eval",
    "tune": "boxmot.engine.commands.tune",
    "research": "boxmot.engine.commands.research",
    "train-reid": "boxmot.engine.commands.reid.train",
    "eval-reid": "boxmot.engine.commands.reid.evaluate",
    "compare-reid": "boxmot.engine.commands.reid.compare",
    "export": "boxmot.engine.commands.reid.export",
    "build": "boxmot.engine.commands.build",
}

_STANDALONE_REID_RUNTIME_BY_COMMAND = {
    "boxmot.engine.commands.reid.ablation": "boxmot.reid.training.ablation_runs",
    "boxmot.engine.commands.reid.human_pretrain": "boxmot.reid.training.human_pretraining_runner",
    "boxmot.engine.commands.reid.privileged_cache": "boxmot.reid.training.privileged_cache",
    "boxmot.engine.commands.reid.teacher_extract": "boxmot.reid.training.teacher_extraction",
}


def _imported_modules(module_name: str, blocked_modules: tuple[str, ...]) -> list[str]:
    probe = (
        "import importlib, json, sys; "
        "module_name = sys.argv[1]; "
        "blocked = json.loads(sys.argv[2]); "
        "importlib.import_module(module_name); "
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            module_name,
            json.dumps(blocked_modules),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _imported_heavy_modules(module_name: str) -> list[str]:
    return _imported_modules(module_name, _HEAVY_RUNTIME_MODULES)


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.reid.training",
        "boxmot.engine.config",
        "boxmot.engine.cli",
    ),
)
def test_startup_modules_keep_ml_runtimes_lazy(module_name: str):
    assert _imported_heavy_modules(module_name) == []


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.datasets",
        "boxmot.datasets.config",
        "boxmot.detectors",
        "boxmot.detectors.config",
        "boxmot.reid.config",
        "boxmot.engine.experiment_config",
        "boxmot.engine.commands.eval",
    ),
)
def test_evaluation_selectors_keep_data_and_model_runtimes_lazy(module_name: str) -> None:
    """Selecting profiles must not load tensor runtimes before progress can start."""

    assert _imported_modules(module_name, _CONFIG_RUNTIME_MODULES) == []


def test_training_package_preserves_public_exports_without_resolving_them():
    expected_exports = {
        "AdaSPLoss",
        "AugmentationConfig",
        "BaseTrainer",
        "CenterLoss",
        "CrossEntropyLabelSmooth",
        "CrossScaleMajorityMarginLoss",
        "DataConfig",
        "EvalConfig",
        "LossConfig",
        "METRIC_LOSS_REGISTRY",
        "ModelConfig",
        "MultiSimilarityLoss",
        "OptimizationConfig",
        "ReIDTrainConfig",
        "RunConfig",
        "TreeBoostAPLoss",
        "TripletLoss",
        "WeightedRegularizedTripletLoss",
        "evaluate_ranking",
    }
    probe = "import json; import boxmot.reid.training as training; print(json.dumps(sorted(training.__all__)))"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert set(json.loads(completed.stdout)) == expected_exports


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.engine.eval.replay",
        "boxmot.engine.eval.evaluator",
        "boxmot.native",
    ),
)
def test_build_replay_imports_do_not_load_model_runtimes(module_name: str):
    assert _imported_modules(module_name, _MODEL_RUNTIME_MODULES) == []


def test_evaluator_import_stays_light_until_the_workflow_panel_starts():
    blocked_modules = (*_MODEL_RUNTIME_MODULES, "gdown", "pandas")
    assert _imported_modules("boxmot.engine.eval.evaluator", blocked_modules) == []


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.detectors.config",
        "boxmot.reid.config",
        "boxmot.segmentors.config",
    ),
)
def test_component_selector_modules_keep_ultralytics_runtime_lazy(module_name: str):
    assert (
        _imported_modules(
            module_name,
            ("ultralytics", "boxmot.detectors.backends.ultralytics"),
        )
        == []
    )


def test_engine_command_namespace_does_not_eagerly_import_children():
    assert _imported_modules("boxmot.engine.commands", _REGISTERED_COMMAND_MODULES) == []


def test_reid_command_namespace_does_not_eagerly_import_children():
    assert _imported_modules("boxmot.engine.commands.reid", _REID_COMMAND_MODULES) == []


def test_cli_help_does_not_load_model_runtimes():
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "sys.argv = ['boxmot', '--help']; "
        "\ntry:\n runpy.run_module('boxmot.engine.cli', run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_MODEL_RUNTIME_MODULES)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []


def test_cli_help_does_not_load_domain_runtime_packages():
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "sys.argv = ['boxmot', '--help']; "
        "\ntry:\n runpy.run_module('boxmot.engine.cli', run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_ROOT_HELP_RUNTIME_MODULES)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []


def test_cli_help_does_not_resolve_command_adapters():
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "sys.argv = ['boxmot', '--help']; "
        "\ntry:\n runpy.run_module('boxmot.engine.cli', run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_REGISTERED_COMMAND_MODULES)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []


@pytest.mark.parametrize(("command_name", "expected_module"), _COMMAND_MODULE_BY_NAME.items())
def test_command_help_resolves_only_selected_adapter(command_name: str, expected_module: str):
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "command_name = sys.argv[2]; "
        "sys.argv = ['boxmot', command_name, '--help']; "
        "\ntry:\n runpy.run_module('boxmot.engine.cli', run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_REGISTERED_COMMAND_MODULES), command_name],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == [expected_module]


@pytest.mark.parametrize(
    "command_name",
    (
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
    ),
)
def test_command_help_keeps_model_runtimes_lazy(command_name: str):
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "command_name = sys.argv[2]; "
        "sys.argv = ['boxmot', command_name, '--help']; "
        "\ntry:\n runpy.run_module('boxmot.engine.cli', run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_MODEL_RUNTIME_MODULES), command_name],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []


def test_eval_help_keeps_data_and_model_runtimes_lazy() -> None:
    """Help exercises Click's complete command loading path in a fresh process."""

    probe = (
        "import json, sys; "
        "from click.testing import CliRunner; "
        "from boxmot.engine.cli import boxmot; "
        "result = CliRunner().invoke(boxmot, ['eval', '--help']); "
        "assert result.exit_code == 0, result.output; "
        "assert '--detector' in result.output; "
        "blocked = json.loads(sys.argv[1]); "
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(_CONFIG_RUNTIME_MODULES)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    ("module_name", "runtime_module"),
    _STANDALONE_REID_RUNTIME_BY_COMMAND.items(),
)
def test_standalone_reid_help_keeps_domain_runtime_lazy(module_name: str, runtime_module: str):
    blocked_modules = (*_HEAVY_RUNTIME_MODULES, runtime_module)
    probe = (
        "import json, runpy, sys; "
        "blocked = json.loads(sys.argv[1]); "
        "module_name = sys.argv[2]; "
        "sys.argv = [module_name, '--help']; "
        "\ntry:\n runpy.run_module(module_name, run_name='__main__')\n"
        "except SystemExit:\n pass\n"
        "print(json.dumps([name for name in blocked if name in sys.modules]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(blocked_modules), module_name],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == []
