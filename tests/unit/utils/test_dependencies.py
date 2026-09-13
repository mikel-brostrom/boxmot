"""Metadata-only dependency checks for installed wheels and editable projects."""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from email.message import Message
from typing import Callable

import pytest
from packaging.utils import canonicalize_name

import boxmot.utils.dependencies as dependencies
from boxmot.utils.dependencies import (
    MissingDependencyError,
    extra_requirements,
    missing_requirements,
    require_extra,
    require_packages,
    requirement_satisfied,
)


@dataclass
class InstalledDistribution:
    """Only the installed metadata needed by dependency inspection."""

    version: str
    requires: tuple[str, ...] = ()
    metadata: Message = field(default_factory=Message)


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Model independent installed distributions without touching the environment."""
    packages: dict[str, InstalledDistribution] = {}

    def distribution(name: str) -> InstalledDistribution:
        try:
            return packages[canonicalize_name(name)]
        except KeyError as exc:
            raise dependencies.PackageNotFoundError(name) from exc

    def add(name: str, version: str = "1.0", *, requires: tuple[str, ...] = (), extras: tuple[str, ...] = ()) -> None:
        metadata = Message()
        for extra in extras:
            metadata["Provides-Extra"] = extra
        packages[canonicalize_name(name)] = InstalledDistribution(version, requires, metadata)

    monkeypatch.setattr(dependencies, "distribution", distribution)
    return add


def test_inactive_requirement_never_looks_up_a_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_lookup(name: str) -> None:
        raise AssertionError(f"Unexpected metadata lookup for {name}")

    monkeypatch.setattr(dependencies, "distribution", reject_lookup)
    assert requirement_satisfied("missing-package>=1; python_version < '0'")
    assert requirement_satisfied("missing-package @ https://example.com/package.whl ; python_version < '0'")


def test_missing_requirements_preserves_order_and_version_constraints(installed: Callable[..., None]) -> None:
    installed("demo-package", "1.2.0")

    assert missing_requirements(
        ("demo-package>=1.0", "demo-package>=2.0", "missing-package", "missing-package")
    ) == ["demo-package>=2.0", "missing-package"]


def test_base_package_does_not_satisfy_its_requested_extra(installed: Callable[..., None]) -> None:
    installed("ray", "2.52.1", requires=("tensorboardX>=1.9; extra == 'tune'",), extras=("tune",))

    assert requirement_satisfied("ray==2.52.1")
    assert not requirement_satisfied("ray[tune]==2.52.1")
    installed("tensorboardX", "1.8")
    assert not requirement_satisfied("ray[tune]==2.52.1")
    installed("tensorboardX", "1.9")
    assert requirement_satisfied("ray[tune]==2.52.1")


def test_nested_extras_and_their_transitive_dependencies_are_required(installed: Callable[..., None]) -> None:
    installed("gepa", "0.1.1", requires=("runner[cli]>=2; extra == 'full'",), extras=("full",))
    installed("runner", "2.0", requires=("click>=8; extra == 'cli'",), extras=("cli",))
    installed("click", "8.1", requires=("terminal>=1",))

    assert not requirement_satisfied("gepa[full]>=0.1.1")
    installed("terminal")
    assert requirement_satisfied("gepa[full]>=0.1.1")


def test_only_active_extras_and_platform_requirements_are_checked(installed: Callable[..., None]) -> None:
    installed(
        "demo",
        requires=(
            "base>=1",
            "selected; extra == 'feature-a'",
            "unselected; extra == 'feature-b'",
            "impossible; python_version < '0' and extra == 'feature-a'",
            "unused @ https://example.com/unused.whl ; extra == 'feature-b'",
        ),
        extras=("feature-a", "feature-b"),
    )
    installed("base")
    installed("selected")

    assert requirement_satisfied("demo[Feature_A]")
    assert not requirement_satisfied("demo[feature-b]")
    assert not requirement_satisfied("demo[undeclared]")


def test_dependency_cycles_terminate_and_still_check_each_constraint(installed: Callable[..., None]) -> None:
    installed("a", requires=("b",))
    installed("b", requires=("a>=1",))
    assert requirement_satisfied("a")

    installed("b", requires=("a>=2",))
    assert not requirement_satisfied("a")


def test_dependency_cycles_do_not_hide_missing_siblings(installed: Callable[..., None]) -> None:
    installed("a", requires=("b", "missing"))
    installed("b", requires=("a",))

    assert missing_requirements(("a", "b")) == ["a", "b"]


def test_inspection_does_not_cache_results_across_installations(installed: Callable[..., None]) -> None:
    assert missing_requirements(("new-package>=2",)) == ["new-package>=2"]
    installed("new-package", "2.0")
    assert missing_requirements(("new-package>=2",)) == []


def test_extra_requirements_use_installed_metadata_and_resolve_markers(installed: Callable[..., None]) -> None:
    installed(
        "boxmot",
        "25.0.0",
        requires=(
            "base>=1",
            "ray[tune]==2.52.1; extra == 'evolve'",
            "unselected; extra == 'research'",
            "impossible; python_version < '0' and extra == 'evolve'",
            "base>=1; extra == 'evolve'",
        ),
        extras=("evolve", "research", "empty"),
    )

    assert extra_requirements("evolve") == ("base>=1", "ray[tune]==2.52.1")
    assert extra_requirements("empty") == ("base>=1",)
    assert missing_requirements(extra_requirements("evolve")) == ["base>=1", "ray[tune]==2.52.1"]


def test_extra_names_are_normalized_and_validated(installed: Callable[..., None]) -> None:
    installed("boxmot", extras=("feature-a",))
    assert extra_requirements("Feature_A") == ()

    with pytest.raises(ValueError, match="Unknown boxmot extra"):
        extra_requirements("unknown")
    with pytest.raises(ValueError, match="extra name"):
        extra_requirements("")


def test_missing_distribution_metadata_has_installation_guidance(installed: Callable[..., None]) -> None:
    with pytest.raises(MissingDependencyError, match="Installed metadata") as error:
        extra_requirements("onnx")

    assert shlex.join([sys.executable, "-m", "pip", "install", "boxmot"]) in str(error.value)


def test_satisfied_extra_does_not_require_an_installer(installed: Callable[..., None]) -> None:
    installed("boxmot", requires=("openvino>=2025.2; extra == 'openvino'",), extras=("openvino",))
    installed("openvino", "2025.2")

    require_extra("openvino")
    require_extra("openvino")


def test_missing_extra_reports_explicit_interpreter_bound_install(installed: Callable[..., None]) -> None:
    installed("boxmot", requires=("gepa[full]>=0.1.1; extra == 'research'",), extras=("research",))
    installed("gepa", "0.1.1", requires=("runner; extra == 'full'",), extras=("full",))

    with pytest.raises(MissingDependencyError, match="gepa") as error:
        require_extra("research", purpose="Tracker research")

    message = str(error.value)
    assert "Tracker research" in message
    assert shlex.split(message.splitlines()[-1]) == [
        sys.executable, "-m", "boxmot.engine.cli", "install", "--extra", "research"
    ]


def test_missing_packages_preserve_index_arguments_in_installation_guidance(installed: Callable[..., None]) -> None:
    with pytest.raises(MissingDependencyError) as error:
        require_packages(
            ("nvidia-tensorrt>=1",),
            purpose="TensorRT inference",
            extra_args=("--extra-index-url", "https://pypi.ngc.nvidia.com"),
        )

    assert shlex.split(str(error.value).splitlines()[-1]) == [
        sys.executable, "-m", "boxmot.engine.cli", "install", "--requirement", "nvidia-tensorrt>=1",
        "--extra-index-url", "https://pypi.ngc.nvidia.com",
    ]


def test_active_direct_urls_are_not_mistaken_for_installed_index_packages(installed: Callable[..., None]) -> None:
    installed("demo")

    with pytest.raises(ValueError, match="direct URL"):
        requirement_satisfied("demo @ https://example.com/demo.whl")
