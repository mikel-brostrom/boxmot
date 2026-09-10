"""Install optional BoxMOT dependencies through an explicit CLI action."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence

import click


def _installer_commands() -> list[list[str]]:
    """Return installers bound to the Python environment running BoxMOT."""

    commands: list[list[str]] = []
    if shutil.which("uv"):
        commands.append(["uv", "pip", "install", "--python", sys.executable])
    commands.append([sys.executable, "-m", "pip", "install"])
    return commands


def _index_arguments(extra_args: Sequence[str]) -> tuple[str, ...]:
    """Allow additional indices without allowing environment-target overrides."""

    arguments = tuple(extra_args)
    if len(arguments) % 2 or any(
        arguments[index] != "--extra-index-url" or not arguments[index + 1] or arguments[index + 1].startswith("-")
        for index in range(0, len(arguments), 2)
    ):
        raise ValueError("Installation arguments must be '--extra-index-url URL' pairs.")
    return arguments


def install_requirements(
    requirements: Iterable[str],
    *,
    extra_args: Sequence[str] = (),
    verbose: bool = True,
) -> None:
    """Install missing requirements explicitly and verify the resulting environment."""

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    from boxmot.utils import logger as LOGGER
    from boxmot.utils.dependencies import missing_requirements, require_packages

    requested = tuple(dict.fromkeys(requirements))
    if any(canonicalize_name(Requirement(requirement).name) == "boxmot" for requirement in requested):
        raise ValueError("This command installs dependencies. Install or upgrade BoxMOT separately with uv or pip.")
    installer_arguments = _index_arguments(extra_args)
    missing = missing_requirements(requested)
    if not missing:
        return

    # Satisfied constraints can still bound versions of packages the resolver
    # must change, so give each installer the complete requested requirement set.
    commands = [[*installer, *installer_arguments, *requested] for installer in _installer_commands()]
    failures: list[str] = []
    if verbose:
        LOGGER.info("Installing dependencies: %s", ", ".join(missing))
    for index, command in enumerate(commands):
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                (exc.stderr or exc.stdout or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            )
            failures.append(f"{command[0]}: {detail.strip()}")
            if index + 1 < len(commands) and verbose:
                LOGGER.warning("Dependency installer failed; retrying with %s.", commands[index + 1][0])
            continue

        if verbose:
            for output in (completed.stdout, completed.stderr):
                for line in (output or "").splitlines():
                    if line.strip():
                        LOGGER.info(line)
        importlib.invalidate_caches()
        require_packages(requested, purpose="Dependency installation", extra_args=installer_arguments)
        return

    raise RuntimeError("Dependency installation failed.\n" + "\n".join(failures))


def install_extras(
    extras: Iterable[str],
    *,
    requirements: Iterable[str] = (),
    extra_args: Sequence[str] = (),
    verbose: bool = True,
) -> None:
    """Install selected extras' dependencies without reinstalling BoxMOT itself."""

    from packaging.utils import canonicalize_name

    from boxmot.utils.dependencies import extra_requirements

    selected = tuple(dict.fromkeys(extra.strip() for extra in extras))
    for extra in selected:
        profile = canonicalize_name(extra)
        if profile in {"cpu", "cu130"}:
            raise ValueError(
                f"The '{profile}' extra selects a PyTorch wheel profile. "
                f"Use the documented 'uv sync --extra {profile}' workflow from a BoxMOT source checkout."
            )
    requested = [requirement for extra in selected for requirement in extra_requirements(extra)]
    requested.extend(requirements)
    install_requirements(requested, extra_args=extra_args, verbose=verbose)


@click.command(help="Install optional dependencies in the active Python environment.")
@click.option(
    "--extra", "extras", multiple=True, metavar="NAME", help="BoxMOT extra to install; repeat for several extras."
)
@click.option(
    "--requirement",
    "requirements",
    multiple=True,
    metavar="SPEC",
    help="Package requirement to install; repeat as needed.",
)
@click.option(
    "--extra-index-url",
    "extra_index_urls",
    multiple=True,
    metavar="URL",
    help="Additional package index; repeat as needed.",
)
@click.option("--quiet", is_flag=True, help="Suppress installation progress and success messages.")
def install(
    extras: tuple[str, ...],
    requirements: tuple[str, ...],
    extra_index_urls: tuple[str, ...],
    quiet: bool,
) -> None:
    """Resolve selected requirements and install only missing dependencies."""

    if not extras and not requirements:
        raise click.UsageError("Provide at least one --extra or --requirement.")

    extra_args = tuple(argument for url in extra_index_urls for argument in ("--extra-index-url", url))
    try:
        install_extras(extras, requirements=requirements, extra_args=extra_args, verbose=not quiet)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not quiet:
        click.echo("Requested dependencies are available.")


__all__ = ("install", "install_extras", "install_requirements")
