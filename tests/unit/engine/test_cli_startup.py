"""Import contracts that keep CLI startup independent of ML runtimes."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

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

_ENGINE_CONFIG_MODULES = (
    "boxmot.engine.config.runtime",
    "boxmot.engine.config.experiments",
    "boxmot.engine.config.trackers",
)

_CALIBRATION_MODULES = (
    "boxmot.engine.calibration.cmc",
    "boxmot.engine.calibration.kalman",
    "boxmot.engine.calibration.kalman_artifacts",
    "boxmot.engine.calibration.kalman_data",
    "boxmot.engine.calibration.kalman_fit",
    "boxmot.engine.calibration.kalman_model",
    "boxmot.engine.calibration.kalman_selection",
)

_ROOT_HELP_RUNTIME_MODULES = (
    "cv2",
    "numpy",
    "torch",
    "ultralytics",
    "yaml",
    "boxmot.detectors",
    "boxmot.trackers.common.factory",
    "boxmot.trackers.common.registry",
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
    "boxmot.engine.commands.segmentors.export",
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
    "export-edgetam": "boxmot.engine.commands.segmentors.export",
    "build": "boxmot.engine.commands.build",
}

_STANDALONE_REID_RUNTIME_BY_COMMAND = {
    "boxmot.engine.commands.reid.ablation": "boxmot.reid.training.ablation_runs",
    "boxmot.engine.commands.reid.human_pretrain": "boxmot.reid.training.human_pretraining_runner",
    "boxmot.engine.commands.reid.privileged_cache": "boxmot.reid.training.privileged_cache",
    "boxmot.engine.commands.reid.teacher_extract": "boxmot.reid.training.teacher_extraction",
}


_ImportProbe = Callable[[str, tuple[str, ...]], list[str]]
_HelpProbe = Callable[[str, tuple[str, ...], tuple[str, ...]], list[str]]


@pytest.fixture(scope="module")
def imported_modules() -> _ImportProbe:
    """Share immutable import snapshots while isolating every target module."""
    snapshots: dict[str, frozenset[str]] = {}
    probe = (
        "import importlib, json, sys; "
        "module_name = sys.argv[1]; "
        "importlib.import_module(module_name); "
        "print(json.dumps(list(sys.modules)))"
    )

    def check_imports(module_name: str, blocked_modules: tuple[str, ...]) -> list[str]:
        """Filter a fresh-process snapshot using each test's import contract."""
        if module_name not in snapshots:
            completed = subprocess.run(
                [sys.executable, "-c", probe, module_name],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            snapshots[module_name] = frozenset(json.loads(completed.stdout))
        return [name for name in blocked_modules if name in snapshots[module_name]]

    return check_imports


@pytest.fixture(scope="module")
def help_imported_modules() -> _HelpProbe:
    """Reuse help snapshots for assertions about the same isolated command."""
    snapshots: dict[tuple[str, tuple[str, ...]], frozenset[str]] = {}
    probe = (
        "import json, runpy, sys; "
        "module_name = sys.argv[1]; "
        "sys.argv = [module_name, *json.loads(sys.argv[2])]; "
        "\ntry:\n runpy.run_module(module_name, run_name='__main__')\n"
        "except SystemExit as error:\n if error.code:\n  raise\n"
        "print(json.dumps(list(sys.modules)))"
    )

    def check_imports(module_name: str, arguments: tuple[str, ...], blocked_modules: tuple[str, ...]) -> list[str]:
        """Run each distinct help command once, retaining successful exit checks."""
        key = (module_name, arguments)
        if key not in snapshots:
            completed = subprocess.run(
                [sys.executable, "-c", probe, module_name, json.dumps(arguments)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            snapshots[key] = frozenset(json.loads(completed.stdout.splitlines()[-1]))
        return [name for name in blocked_modules if name in snapshots[key]]

    return check_imports


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.reid.training",
        "boxmot.engine.config",
        "boxmot.engine.config.runtime",
        "boxmot.engine.cli",
    ),
)
def test_startup_modules_keep_ml_runtimes_lazy(module_name: str, imported_modules: _ImportProbe) -> None:
    assert imported_modules(module_name, _HEAVY_RUNTIME_MODULES) == []


def test_engine_config_namespace_does_not_eagerly_import_children(imported_modules: _ImportProbe) -> None:
    """Loading configuration names must not resolve profiles or runtime defaults."""

    blocked_modules = (*_ENGINE_CONFIG_MODULES, *_CONFIG_RUNTIME_MODULES, "yaml")
    assert imported_modules("boxmot.engine.config", blocked_modules) == []


def test_calibration_namespace_does_not_eagerly_import_children(imported_modules: _ImportProbe) -> None:
    """Importing the namespace must not load data, filter, or model runtimes."""

    assert imported_modules("boxmot.engine.calibration", (*_CALIBRATION_MODULES, *_CONFIG_RUNTIME_MODULES)) == []


@pytest.mark.parametrize("module_name", _CALIBRATION_MODULES)
def test_calibration_imports_do_not_load_search_dependencies(module_name: str, imported_modules: _ImportProbe) -> None:
    """Evaluation can calibrate Kalman filters without installing search extras."""

    assert imported_modules(module_name, ("boxmot.engine.tuning", "ray", "optuna", "hyperopt")) == []


def test_tuner_keeps_sensor_evaluation_and_search_runtimes_lazy(imported_modules: _ImportProbe) -> None:
    """The shared tuner must load its sensor runtime only after selecting sensor inputs."""
    assert (
        imported_modules(
            "boxmot.engine.tuning.tuner",
            ("optuna", "ray", "boxmot.engine.eval.eagermot_kitti"),
        )
        == []
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.datasets",
        "boxmot.datasets.config",
        "boxmot.detectors",
        "boxmot.detectors.config",
        "boxmot.reid.config",
        "boxmot.engine.config.experiments",
        "boxmot.engine.commands.eval",
        "boxmot.engine.commands.track",
    ),
)
def test_evaluation_selectors_keep_data_and_model_runtimes_lazy(
    module_name: str, imported_modules: _ImportProbe
) -> None:
    """Selecting profiles must not load tensor runtimes before progress can start."""

    assert imported_modules(module_name, _CONFIG_RUNTIME_MODULES) == []


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
def test_build_replay_imports_do_not_load_model_runtimes(module_name: str, imported_modules: _ImportProbe) -> None:
    assert imported_modules(module_name, _MODEL_RUNTIME_MODULES) == []


def test_evaluator_import_stays_light_until_the_workflow_panel_starts(imported_modules: _ImportProbe) -> None:
    blocked_modules = (*_MODEL_RUNTIME_MODULES, "gdown", "pandas")
    assert imported_modules("boxmot.engine.eval.evaluator", blocked_modules) == []


@pytest.mark.parametrize(
    "module_name",
    (
        "boxmot.detectors.config",
        "boxmot.reid.config",
        "boxmot.segmentors.config",
    ),
)
def test_component_selector_modules_keep_ultralytics_runtime_lazy(
    module_name: str, imported_modules: _ImportProbe
) -> None:
    assert (
        imported_modules(
            module_name,
            ("ultralytics", "boxmot.detectors.backends.ultralytics"),
        )
        == []
    )


def test_engine_command_namespace_does_not_eagerly_import_children(imported_modules: _ImportProbe) -> None:
    assert imported_modules("boxmot.engine.commands", _REGISTERED_COMMAND_MODULES) == []


def test_reid_command_namespace_does_not_eagerly_import_children(imported_modules: _ImportProbe) -> None:
    assert imported_modules("boxmot.engine.commands.reid", _REID_COMMAND_MODULES) == []


def test_cli_help_does_not_load_model_runtimes(help_imported_modules: _HelpProbe) -> None:
    assert help_imported_modules("boxmot.engine.cli", ("--help",), _MODEL_RUNTIME_MODULES) == []


def test_cli_help_does_not_load_domain_runtime_packages(help_imported_modules: _HelpProbe) -> None:
    assert help_imported_modules("boxmot.engine.cli", ("--help",), _ROOT_HELP_RUNTIME_MODULES) == []


def test_cli_help_does_not_resolve_command_adapters(help_imported_modules: _HelpProbe) -> None:
    assert help_imported_modules("boxmot.engine.cli", ("--help",), _REGISTERED_COMMAND_MODULES) == []


@pytest.mark.parametrize(("command_name", "expected_module"), _COMMAND_MODULE_BY_NAME.items())
def test_command_help_resolves_only_selected_adapter(
    command_name: str, expected_module: str, help_imported_modules: _HelpProbe
) -> None:
    assert help_imported_modules("boxmot.engine.cli", (command_name, "--help"), _REGISTERED_COMMAND_MODULES) == [
        expected_module
    ]


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
        "export-edgetam",
        "build",
    ),
)
def test_command_help_keeps_model_runtimes_lazy(command_name: str, help_imported_modules: _HelpProbe) -> None:
    assert help_imported_modules("boxmot.engine.cli", (command_name, "--help"), _MODEL_RUNTIME_MODULES) == []


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
def test_standalone_reid_help_keeps_domain_runtime_lazy(
    module_name: str, runtime_module: str, help_imported_modules: _HelpProbe
) -> None:
    blocked_modules = (*_HEAVY_RUNTIME_MODULES, runtime_module)
    assert help_imported_modules(module_name, ("--help",), blocked_modules) == []
