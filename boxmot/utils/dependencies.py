"""Inspect installed dependency metadata without importing or installing packages."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable, Sequence
from importlib.metadata import Distribution, PackageNotFoundError, distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


class MissingDependencyError(ImportError):
    """An operation needs dependencies that are unavailable in this interpreter."""


def _active(requirement: Requirement, extras: Iterable[str] = ()) -> bool:
    """Evaluate a dependency for the base distribution and its requested extras."""
    return requirement.marker is None or any(
        requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)
    )


def _provided_extras(installed: Distribution) -> frozenset[str]:
    """Return the normalized extras declared by one installed distribution."""
    return frozenset(canonicalize_name(extra) for extra in installed.metadata.get_all("Provides-Extra", ()))


class _InstalledRequirements:
    """Cache metadata for one inspection while keeping later installs observable."""

    def __init__(self) -> None:
        self._distributions: dict[str, Distribution | None] = {}

    def _distribution(self, name: str) -> Distribution | None:
        name = canonicalize_name(name)
        if name not in self._distributions:
            try:
                self._distributions[name] = distribution(name)
            except PackageNotFoundError:
                self._distributions[name] = None
        return self._distributions[name]

    def satisfied(self, requirement: Requirement, visited: set[tuple[str, frozenset[str]]]) -> bool:
        """Check versions and the active dependency graph, including nested extras."""
        if requirement.url is not None:
            raise ValueError(
                "Dependency checks support package names, extras, and version specifiers. "
                "Install direct URL requirements explicitly with pip or uv."
            )
        installed = self._distribution(requirement.name)
        if installed is None or not requirement.specifier.contains(installed.version, prereleases=True):
            return False

        extras = frozenset(canonicalize_name(extra) for extra in requirement.extras)
        if not extras <= _provided_extras(installed):
            return False

        # Check each incoming version constraint before breaking dependency cycles.
        key = (canonicalize_name(requirement.name), extras)
        if key in visited:
            return True
        visited.add(key)

        for raw in installed.requires or ():
            child = Requirement(raw)
            if _active(child, extras) and not self.satisfied(child, visited):
                return False
        return True


def requirement_satisfied(requirement: str) -> bool:
    """Check an active requirement and its dependency closure in this interpreter.

    Environment markers are respected. Requested extras must be declared and
    their dependencies must also be installed at compatible versions. Package
    inspection uses distribution metadata and never imports the runtime.
    """
    parsed = Requirement(requirement)
    return not _active(parsed) or _InstalledRequirements().satisfied(parsed, set())


def missing_requirements(requirements: Iterable[str]) -> list[str]:
    """Return unsatisfied active requirements in input order, without duplicates."""
    installed = _InstalledRequirements()
    missing: list[str] = []
    for raw in dict.fromkeys(requirements):
        requirement = Requirement(raw)
        if _active(requirement) and not installed.satisfied(requirement, set()):
            missing.append(raw)
    return missing


def extra_requirements(extra: str, *, package: str = "boxmot") -> tuple[str, ...]:
    """Read active base and extra dependencies from installed distribution metadata.

    Editable and wheel installations use the same metadata. Markers are
    evaluated here, then removed so an extra-only requirement remains active
    when checked or passed to an installer on its own.
    """
    if not isinstance(extra, str) or not extra.strip():
        raise ValueError("An extra name must be provided.")
    extra = canonicalize_name(extra.strip())
    try:
        installed = distribution(package)
    except PackageNotFoundError as exc:
        command = shlex.join([sys.executable, "-m", "pip", "install", package])
        raise MissingDependencyError(
            f"Installed metadata for {package!r} is unavailable. Install the package first: {command}"
        ) from exc

    available = _provided_extras(installed)
    if extra not in available:
        raise ValueError(f"Unknown {package} extra {extra!r}. Available extras: {', '.join(sorted(available))}.")

    requirements: list[str] = []
    for raw in installed.requires or ():
        requirement = Requirement(raw)
        if _active(requirement, (extra,)):
            requirement.marker = None
            requirements.append(str(requirement))
    return tuple(dict.fromkeys(requirements))


def _install_command(
    requirements: Iterable[str] = (), *, extras: Iterable[str] = (), extra_args: Sequence[str] = ()
) -> str:
    """Describe an explicit install using the interpreter that needs the packages."""
    command = [sys.executable, "-m", "boxmot.engine.cli", "install"]
    for extra in extras:
        command.extend(("--extra", extra))
    for requirement in requirements:
        command.extend(("--requirement", requirement))
    command.extend(extra_args)
    return shlex.join(command)


def require_packages(
    requirements: Iterable[str], *, purpose: str = "This operation", extra_args: Sequence[str] = ()
) -> None:
    """Raise an actionable error if packages are missing; never install them."""
    missing = missing_requirements(requirements)
    if missing:
        command = _install_command(missing, extra_args=extra_args)
        raise MissingDependencyError(
            f"{purpose} requires missing or incompatible dependencies: {', '.join(missing)}. "
            f"Install them before retrying:\n{command}"
        )


def require_extra(extra: str, *, purpose: str = "This operation") -> None:
    """Validate a BoxMOT extra without changing the current environment."""
    missing = missing_requirements(extra_requirements(extra))
    if missing:
        raise MissingDependencyError(
            f"{purpose} requires missing or incompatible dependencies for the {extra!r} extra: "
            f"{', '.join(missing)}. Install them before retrying:\n{_install_command(extras=(extra,))}"
        )


__all__ = (
    "MissingDependencyError",
    "extra_requirements",
    "missing_requirements",
    "require_extra",
    "require_packages",
    "requirement_satisfied",
)
