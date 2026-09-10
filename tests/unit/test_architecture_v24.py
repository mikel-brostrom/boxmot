"""Static dependency-direction checks for the v24 package architecture."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "boxmot"

DEPENDENCY_DIRECTION_EXCEPTIONS = frozenset(
    {
        # Resource downloads reuse shared metadata-only dependency validation;
        # package installation remains engine-owned.
        (
            "boxmot/resources/download.py",
            "boxmot.utils.dependencies",
        ),
        # Appearance-enabled tracker adapters intentionally own lazy ReID
        # inference when detections arrive without precomputed embeddings.
        (
            "boxmot/trackers/common/appearance/live.py",
            "boxmot.reid.core",
        ),
        (
            "boxmot/trackers/common/appearance/live.py",
            "boxmot.reid.factory",
        ),
        (
            "boxmot/trackers/common/appearance/live.py",
            "boxmot.reid.protocols",
        ),
        (
            "boxmot/trackers/common/appearance/live.py",
            "boxmot.reid.specs",
        ),
        # The optional tracker-owned ReID protocol names the immutable encoder
        # specification accepted by its public configuration boundary.
        (
            "boxmot/trackers/common/protocols.py",
            "boxmot.reid.specs",
        ),
    }
)

REMOVED_V24_MODULES = (
    "boxmot.api",
    "boxmot.motion",
    "boxmot.core.box_schema",
    "boxmot.data",
    "boxmot.detectors._factory_registry",
    "boxmot.detectors.registry",
    "boxmot.engine.runner",
    "boxmot.engine.sinks",
    "boxmot.engine.sources",
    "boxmot.engine.eval.cache",
    "boxmot.engine.experiments",
    "boxmot.engine.experiment_config",
    "boxmot.engine.tracker_config",
    "boxmot.engine.dataset_resources",
    "boxmot.engine.frame_timing",
    "boxmot.engine.logging",
    "boxmot.engine.dependencies",
    "boxmot.engine.presentation",
    "boxmot.engine.reid",
    "boxmot.engine.commands.reid.base",
    "boxmot.engine.tracking.results",
    "boxmot.engine.tracking.setup_timing",
    "boxmot.engine.tuning.ground_truth_noise",
    "boxmot.engine.tuning.kalman",
    "boxmot.engine.tuning.kalman_data",
    "boxmot.engine.tuning.kalman_model",
    "boxmot.engine.workflows",
    "boxmot.reid._factory_registry",
    "boxmot.segmentors.registry",
    "boxmot.trackers.bbox",
    "boxmot.trackers.hybrid",
    "boxmot.trackers.common.track_models",
    "boxmot.trackers.results",
    "boxmot.pipelines.results",
    "boxmot.engine.artifacts",
    "boxmot.engine.builds",
    "boxmot.engine.catalog",
    "boxmot.engine.component_specs",
    "boxmot.engine.dataset_config",
    "boxmot.engine.experiment",
    "boxmot.native.registry",
    "boxmot.utils.artifact_identity",
    "boxmot.utils.callbacks",
    "boxmot.utils.download",
    "boxmot.utils.misc",
    "boxmot.utils.rich",
    "boxmot.utils.timing",
)

REMOVED_V24_FILES = (
    "api/__init__.py",
    "core/box_schema.py",
    "data/__init__.py",
    "detectors/_factory_registry.py",
    "detectors/registry.py",
    "engine/runner.py",
    "engine/sinks.py",
    "engine/sources.py",
    "engine/eval/cache.py",
    "engine/config.py",
    "engine/experiment_config.py",
    "engine/tracker_config.py",
    "engine/dataset_resources.py",
    "engine/frame_timing.py",
    "engine/logging.py",
    "engine/dependencies.py",
    "engine/experiments/__init__.py",
    "engine/experiments/artifacts.py",
    "engine/experiments/builds.py",
    "engine/experiments/components.py",
    "engine/experiments/experiment.py",
    "engine/presentation/__init__.py",
    "engine/presentation/reporting.py",
    "engine/presentation/results.py",
    "engine/reid/__init__.py",
    "engine/commands/reid/base.py",
    "engine/tracking/results.py",
    "engine/tracking/setup_timing.py",
    "engine/tuning/ground_truth_noise.py",
    "engine/tuning/kalman.py",
    "engine/tuning/kalman_data.py",
    "engine/tuning/kalman_model.py",
    "engine/workflows/__init__.py",
    "engine/workflows/materialize_dataset.py",
    "engine/workflows/reid_ablation.py",
    "engine/workflows/reid_base.py",
    "engine/workflows/reid_compare.py",
    "engine/workflows/reid_evaluate.py",
    "engine/workflows/reid_export.py",
    "engine/workflows/reid_human_pretrain.py",
    "engine/workflows/reid_privileged_cache.py",
    "engine/workflows/reid_teacher_extract.py",
    "engine/workflows/reid_train.py",
    "engine/workflows/reporting.py",
    "engine/workflows/results.py",
    "reid/_factory_registry.py",
    "segmentors/registry.py",
    "trackers/bbox/__init__.py",
    "trackers/bbox/boosttrack.py",
    "trackers/bbox/botsort.py",
    "trackers/bbox/bytetrack.py",
    "trackers/bbox/deepocsort.py",
    "trackers/bbox/hybridsort.py",
    "trackers/bbox/occluboost.py",
    "trackers/bbox/ocsort.py",
    "trackers/bbox/sfsort.py",
    "trackers/bbox/strongsort.py",
    "trackers/hybrid/__init__.py",
    "trackers/hybrid/base.py",
    "trackers/mask/base.py",
    "trackers/common/track_models/__init__.py",
    "trackers/results.py",
    "pipelines/results.py",
    "engine/artifacts.py",
    "engine/builds.py",
    "engine/catalog.py",
    "engine/component_specs.py",
    "engine/dataset_config.py",
    "engine/experiment.py",
    "native/registry.py",
    "utils/artifact_identity.py",
    "utils/callbacks.py",
    "utils/download.py",
    "utils/misc.py",
    "utils/rich/__init__.py",
    "utils/timing.py",
)

REMOVED_V24_PACKAGES = (
    "boxmot.api",
    "boxmot.motion",
    "boxmot.core",
    "boxmot.data",
    "boxmot.engine.experiments",
    "boxmot.engine.presentation",
    "boxmot.engine.reid",
    "boxmot.engine.workflows",
    "boxmot.trackers.bbox",
    "boxmot.trackers.hybrid",
    "boxmot.trackers.common.track_models",
    "boxmot.utils.rich",
)

REMOVED_V24_DIRECTORIES = (
    "motion",
    "engine/experiments",
    "engine/presentation",
    "engine/workflows",
    "trackers/bbox",
    "trackers/hybrid",
    "trackers/common/track_models",
    "utils/rich",
)

MOVED_ENGINE_MODULES = (
    "boxmot.engine.experiment_config",
    "boxmot.engine.tracker_config",
    "boxmot.engine.dataset_resources",
    "boxmot.engine.frame_timing",
    "boxmot.engine.logging",
    "boxmot.engine.dependencies",
    "boxmot.engine.artifacts",
    "boxmot.engine.builds",
    "boxmot.engine.catalog",
    "boxmot.engine.component_specs",
    "boxmot.engine.dataset_config",
    "boxmot.engine.experiment",
    "boxmot.engine.tuning.ground_truth_noise",
    "boxmot.engine.tuning.kalman",
    "boxmot.engine.tuning.kalman_data",
    "boxmot.engine.tuning.kalman_model",
)

REMOVED_COMPONENT_INFRASTRUCTURE_MODULES = (
    "boxmot.detectors._factory_registry",
    "boxmot.detectors.registry",
    "boxmot.reid._factory_registry",
    "boxmot.segmentors.registry",
    "boxmot.utils.artifact_identity",
)

REMOVED_UTILITY_MODULES = (
    "boxmot.utils.callbacks",
    "boxmot.utils.download",
    "boxmot.utils.misc",
    "boxmot.utils.timing",
)

MOVED_TRACKER_SUPPORT_MODULES = (
    "boxmot.trackers.base",
    "boxmot.trackers.config",
    "boxmot.trackers.factory",
    "boxmot.trackers.protocols",
    "boxmot.trackers.registry",
    "boxmot.trackers.specs",
)

REMOVED_TRACKER_FAMILY_PACKAGES = (
    "boxmot.trackers.box",
    "boxmot.trackers.mask",
    "boxmot.trackers.multimodal",
)

ENGINE_COMMAND_LAYOUT = (
    "engine/commands/__init__.py",
    "engine/commands/_options.py",
    "engine/commands/_support.py",
    "engine/commands/build.py",
    "engine/commands/eval.py",
    "engine/commands/install.py",
    "engine/commands/materialize.py",
    "engine/commands/research.py",
    "engine/commands/sensor_eval.py",
    "engine/commands/sensor_tune.py",
    "engine/commands/time_variant.py",
    "engine/commands/track.py",
    "engine/commands/trackrcnn.py",
    "engine/commands/tune.py",
    "engine/commands/reid/__init__.py",
    "engine/commands/reid/_options.py",
    "engine/commands/reid/ablation.py",
    "engine/commands/reid/compare.py",
    "engine/commands/reid/evaluate.py",
    "engine/commands/reid/export.py",
    "engine/commands/reid/human_pretrain.py",
    "engine/commands/reid/privileged_cache.py",
    "engine/commands/reid/teacher_extract.py",
    "engine/commands/reid/train.py",
)

ENGINE_OWNERSHIP_LAYOUT = (
    "datasets/config.py",
    *ENGINE_COMMAND_LAYOUT,
    "engine/calibration/__init__.py",
    "engine/calibration/ground_truth_noise.py",
    "engine/calibration/kalman.py",
    "engine/calibration/kalman_data.py",
    "engine/calibration/kalman_model.py",
    "engine/config/__init__.py",
    "engine/config/runtime.py",
    "engine/config/experiments.py",
    "engine/config/trackers.py",
    "engine/eval/output.py",
    "engine/materialization/builds.py",
    "engine/materialization/catalog.py",
    "engine/materialization/resources.py",
    "engine/tracking/timing.py",
    "engine/tracking/timestamps.py",
    "engine/tuning/calibration_profile.py",
    "engine/tuning/eagermot_kitti.py",
    "engine/tuning/results.py",
    "engine/ui/__init__.py",
    "engine/ui/logging.py",
    "engine/ui/core/ui.py",
    "engine/ui/reporters/materialize.py",
    "engine/ui/reporters/validation.py",
    "engine/ui/workflow/pipeline.py",
)

DOMAIN_RESOLUTION_LAYOUT = (
    "components/artifacts.py",
    "components/resolution.py",
    "detectors/config.py",
    "reid/config.py",
    "segmentors/config.py",
)

RESOURCE_OWNERSHIP_LAYOUT = (
    "resources/__init__.py",
    "resources/download.py",
    "resources/paths.py",
)

TRACKER_OWNERSHIP_LAYOUT = (
    "trackers/common/base.py",
    "trackers/common/config.py",
    "trackers/common/factory.py",
    "trackers/common/manifest.py",
    "trackers/common/protocols.py",
    "trackers/common/registry.py",
    "trackers/common/specs.py",
    "trackers/common/box/__init__.py",
    "trackers/common/box/base.py",
    "trackers/common/box/geometry.py",
    "trackers/common/motion/__init__.py",
    "trackers/common/motion/models.py",
    "trackers/common/motion/tracker.py",
    "trackers/common/motion/cmc/__init__.py",
    "trackers/common/motion/cmc/base.py",
    "trackers/common/motion/cmc/registry.py",
    "trackers/common/motion/cmc/integration.py",
    "trackers/common/motion/kalman_filters/__init__.py",
    "trackers/common/motion/kalman_filters/base.py",
    "trackers/common/motion/kalman_filters/noise.py",
    "trackers/common/motion/kalman_filters/fitting.py",
    "trackers/boosttrack/__init__.py",
    "trackers/boosttrack/tracker.py",
    "trackers/botsort/__init__.py",
    "trackers/botsort/tracker.py",
    "trackers/bytetrack/__init__.py",
    "trackers/bytetrack/tracker.py",
    "trackers/deepocsort/__init__.py",
    "trackers/deepocsort/tracker.py",
    "trackers/hybridsort/__init__.py",
    "trackers/hybridsort/tracker.py",
    "trackers/occluboost/__init__.py",
    "trackers/occluboost/tracker.py",
    "trackers/ocsort/__init__.py",
    "trackers/ocsort/tracker.py",
    "trackers/sfsort/__init__.py",
    "trackers/sfsort/tracker.py",
    "trackers/strongsort/__init__.py",
    "trackers/strongsort/tracker.py",
    "trackers/eagermot/__init__.py",
    "trackers/eagermot/tracker.py",
    "trackers/maf_hda/__init__.py",
    "trackers/maf_hda/tracker.py",
)


def _absolute_imports(path: Path, package: str) -> set[str]:
    """Return absolute module names imported by one source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module:
                imports.add(node.module)
            continue

        parts = package.split(".")
        base = parts[: max(0, len(parts) - node.level + 1)]
        if node.module:
            base.extend(node.module.split("."))
        if base:
            imports.add(".".join(base))
    return imports


def _containing_package(path: Path) -> str:
    module_parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
    if module_parts[-1] == "__init__":
        module_parts.pop()
    else:
        module_parts.pop()
    return ".".join(module_parts)


def _imports_for_tree(package_name: str) -> dict[Path, set[str]]:
    root = PACKAGE_ROOT.joinpath(*package_name.split(".")[1:])
    return {
        path: _absolute_imports(path, _containing_package(path))
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(f"{forbidden}.")


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        (
            "boxmot.engine.calibration",
            {"boxmot.engine.tuning", "ray", "optuna", "hyperopt"},
        ),
        (
            "boxmot.resources",
            {
                "boxmot.components",
                "boxmot.structures",
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
                "boxmot.native",
                "boxmot.utils",
            },
        ),
        (
            "boxmot.components",
            {
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.utils",
            {
                "boxmot.structures",
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.structures",
            {
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.detectors",
            {
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.segmentors",
            {
                "boxmot.detectors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.reid",
            {
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.trackers",
            {
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
        (
            "boxmot.datasets",
            {
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.engine",
            },
        ),
        ("boxmot.pipelines", {"boxmot.datasets", "boxmot.engine"}),
        ("boxmot.postprocessing", {"boxmot.engine"}),
        (
            "boxmot.native",
            {
                "boxmot.components",
                "boxmot.detectors",
                "boxmot.segmentors",
                "boxmot.reid",
                "boxmot.structures",
                "boxmot.trackers",
                "boxmot.pipelines",
                "boxmot.datasets",
                "boxmot.engine",
            },
        ),
    ],
)
def test_dependency_direction_allows_only_declared_exceptions(package: str, forbidden: set[str]) -> None:
    violations: list[str] = []
    for path, imports in _imports_for_tree(package).items():
        for imported in sorted(imports):
            relative = path.relative_to(PACKAGE_ROOT.parent).as_posix()
            if (
                any(_matches(imported, prefix) for prefix in forbidden)
                and (
                    relative,
                    imported,
                )
                not in DEPENDENCY_DIRECTION_EXCEPTIONS
            ):
                violations.append(f"{relative} imports {imported}")
    assert violations == []


def test_removed_v24_modules_are_physically_absent() -> None:
    """The coordinated public cutover must not leave importable shims behind."""

    present = [relative for relative in REMOVED_V24_FILES if (PACKAGE_ROOT / relative).is_file()]
    assert present == []


def test_removed_v24_directories_are_physically_absent() -> None:
    """Removed package trees must not survive as importable namespace packages."""

    present = [relative for relative in REMOVED_V24_DIRECTORIES if (PACKAGE_ROOT / relative).is_dir()]
    assert present == []


def test_engine_ownership_layout_is_exact() -> None:
    missing = [relative for relative in ENGINE_OWNERSHIP_LAYOUT if not (PACKAGE_ROOT / relative).is_file()]
    assert missing == []


def test_engine_root_contains_only_cli_composition() -> None:
    """Workflow support belongs beside its owning engine subsystem."""

    engine_files = {path.name for path in (PACKAGE_ROOT / "engine").glob("*.py")}
    assert engine_files == {"__init__.py", "cli.py"}


def test_component_resolution_is_owned_by_domain_packages() -> None:
    missing = [relative for relative in DOMAIN_RESOLUTION_LAYOUT if not (PACKAGE_ROOT / relative).is_file()]
    assert missing == []


def test_engine_command_layout_is_exact() -> None:
    command_root = PACKAGE_ROOT / "engine" / "commands"
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in command_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert actual == set(ENGINE_COMMAND_LAYOUT)


def test_engine_command_adapters_do_not_import_cli_composition_root() -> None:
    violations = []
    for relative in ENGINE_COMMAND_LAYOUT:
        path = PACKAGE_ROOT / relative
        package = _containing_package(path)
        if any(_matches(imported, "boxmot.engine.cli") for imported in _absolute_imports(path, package)):
            violations.append(relative)
    assert violations == []


def test_engine_runtime_config_does_not_import_reid_domain() -> None:
    path = PACKAGE_ROOT / "engine" / "config" / "runtime.py"
    imports = _absolute_imports(path, "boxmot.engine.config")
    assert not any(_matches(imported, "boxmot.reid") for imported in imports)


def test_resource_ownership_layout_is_exact() -> None:
    missing = [relative for relative in RESOURCE_OWNERSHIP_LAYOUT if not (PACKAGE_ROOT / relative).is_file()]
    assert missing == []


def test_tracker_shared_support_and_algorithm_ownership_is_present() -> None:
    missing = [relative for relative in TRACKER_OWNERSHIP_LAYOUT if not (PACKAGE_ROOT / relative).is_file()]
    assert missing == []


def test_utils_contains_only_cross_domain_helpers() -> None:
    utility_files = {path.name for path in (PACKAGE_ROOT / "utils").glob("*.py")}
    assert utility_files == {"__init__.py", "config.py", "dependencies.py", "devices.py"}


def test_dependency_validation_has_no_installer_or_legacy_checker_module() -> None:
    """Runtime validation stays separate from engine-owned installation."""

    dependency_module = PACKAGE_ROOT / "utils" / "dependencies.py"
    imports = _absolute_imports(dependency_module, "boxmot.utils")

    assert not any(_matches(imported, "subprocess") or _matches(imported, "boxmot") for imported in imports)
    assert not (PACKAGE_ROOT / "utils" / "checks.py").exists()
    assert importlib.util.find_spec("boxmot.utils.checks") is None


@pytest.mark.parametrize("module_name", MOVED_ENGINE_MODULES)
def test_moved_engine_modules_are_not_importable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize("module_name", REMOVED_COMPONENT_INFRASTRUCTURE_MODULES)
def test_replaced_component_infrastructure_is_not_importable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize("module_name", REMOVED_UTILITY_MODULES)
def test_moved_utility_modules_are_not_importable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    ("relative", "package"),
    [
        ("engine/calibration/__init__.py", "boxmot.engine.calibration"),
        ("engine/config/__init__.py", "boxmot.engine.config"),
        ("engine/ui/__init__.py", "boxmot.engine.ui"),
        ("engine/ui/reporters/__init__.py", "boxmot.engine.ui.reporters"),
        ("engine/ui/workflow/__init__.py", "boxmot.engine.ui.workflow"),
        ("resources/__init__.py", "boxmot.resources"),
    ],
)
def test_new_engine_namespaces_have_no_eager_child_imports(relative: str, package: str) -> None:
    init_path = PACKAGE_ROOT / relative
    child_imports = {
        imported for imported in _absolute_imports(init_path, package) if imported.startswith(f"{package}.")
    }
    assert child_imports == set()


@pytest.mark.parametrize(
    ("relative", "package"),
    [
        ("engine/commands/__init__.py", "boxmot.engine.commands"),
        ("engine/commands/reid/__init__.py", "boxmot.engine.commands.reid"),
    ],
)
def test_engine_command_namespaces_have_no_eager_child_imports(relative: str, package: str) -> None:
    init_path = PACKAGE_ROOT / relative
    child_imports = {
        imported for imported in _absolute_imports(init_path, package) if imported.startswith(f"{package}.")
    }
    assert child_imports == set()


@pytest.mark.parametrize("module_name", REMOVED_V24_PACKAGES)
def test_removed_v24_packages_are_not_importable(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


def test_source_tree_never_imports_removed_v24_modules() -> None:
    """Reject reverse imports and accidental resurrection of obsolete namespaces."""

    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        package = _containing_package(path)
        for imported in sorted(_absolute_imports(path, package)):
            if any(_matches(imported, prefix) for prefix in REMOVED_V24_MODULES):
                violations.append(f"{path.relative_to(PACKAGE_ROOT.parent)} imports {imported}")
    assert violations == []


def test_source_tree_uses_canonical_tracker_support_and_algorithm_paths() -> None:
    """Reject callers that retain imports from the previous tracker layout."""

    removed_modules = (*MOVED_TRACKER_SUPPORT_MODULES, *REMOVED_TRACKER_FAMILY_PACKAGES)
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        for imported in sorted(_absolute_imports(path, _containing_package(path))):
            if any(_matches(imported, prefix) for prefix in removed_modules):
                violations.append(f"{path.relative_to(PACKAGE_ROOT.parent)} imports {imported}")
    assert violations == []


def test_source_tree_uses_backport_for_typing_self() -> None:
    """Keep workflow imports valid on the declared Python 3.10 minimum."""

    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "typing":
                continue
            if any(alias.name == "Self" for alias in node.names):
                violations.append(f"{path.relative_to(PACKAGE_ROOT.parent)} imports typing.Self")
    assert violations == []


def test_engine_materialization_does_not_own_parquet_implementations() -> None:
    materialization = PACKAGE_ROOT / "engine" / "materialization"
    parquet_imports = []
    for path in materialization.rglob("*.py"):
        for imported in _absolute_imports(path, _containing_package(path)):
            if imported == "pyarrow" or imported.startswith("pyarrow."):
                parquet_imports.append(f"{path.relative_to(PACKAGE_ROOT.parent)} imports {imported}")

    assert parquet_imports == []
    assert not tuple((materialization / "writers").glob("*.py"))
