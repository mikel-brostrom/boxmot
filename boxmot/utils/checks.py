import shlex
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional, Sequence

from packaging.requirements import Requirement

from boxmot.utils import ROOT

# Replace this import with your logger, or use logging.getLogger(__name__)
from boxmot.utils import logger as LOGGER

REQUIREMENTS_FILE = Path("requirements.txt")


def requirement_satisfied(requirement: str) -> bool:
    """Return whether a requirement specifier is satisfied in the active env."""
    req = Requirement(requirement)
    if req.marker is not None and not req.marker.evaluate():
        return True
    try:
        inst_ver = version(req.name)
    except PackageNotFoundError:
        return False
    return not req.specifier or req.specifier.contains(inst_ver, prereleases=True)


def missing_requirements(requirements: Iterable[str]) -> list[str]:
    """Return requirement specifiers from *requirements* that are not satisfied."""
    return [requirement for requirement in requirements if not requirement_satisfied(requirement)]


class RequirementsChecker:
    """
    Runtime dependency helper.

    Features:
      - Check/install a list of requirement specifiers (e.g., ["yolox", "onnx>=1.15"])
      - Read and install from a requirements.txt
      - Install a project *extra* (PEP 621 optional-dependencies)
    """

    def __init__(
        self, group: Optional[str] = None, requirements_file: Path = REQUIREMENTS_FILE
    ):
        """
        If `group` is provided, you *may* choose to call `sync_group_or_extra(group=group)`
        before doing work. Otherwise you can use `check_requirements_file()` or `check_packages()`.
        """
        self.group = group
        self.requirements_file = requirements_file

    # ---------- public API ----------

    def check_packages(
        self,
        requirements: Iterable[str],
        extra_args: Optional[Sequence[str] | str] = None,
    ):
        """
        Check & install packages specified by requirement strings.

        :param requirements: iterable of requirement specifiers as strings
        :param extra_args: extra args for the installer (e.g. ["--upgrade"])
        """
        missing = self._missing_packages(requirements)
        if missing:
            self._install_packages(missing, extra_args)

    def sync_extra(
        self,
        extra: str,
        extra_args: Optional[Sequence[str] | str] = None,
        verbose: bool = True,
    ):
        """
        Install a project *extra* (PEP 621 optional-dependencies).
        - From source checkout: install -e ".[extra]"
        - From PyPI install:    install "boxmot[extra]"
        Uses uv when available, otherwise the active Python's pip.
        """
        if not extra:
            raise ValueError("Extra name must be provided (e.g. 'openvino', 'export').")

        # Skip install if all packages in the extra are already satisfied.
        root_pyproject = ROOT / "pyproject.toml"
        if root_pyproject.is_file():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[no-redef]
                with open(root_pyproject, "rb") as f:
                    pyproject = tomllib.load(f)
                extra_pkgs = (
                    pyproject.get("project", {})
                    .get("optional-dependencies", {})
                    .get(extra, [])
                )
                if extra_pkgs and not self._missing_packages(extra_pkgs):
                    return
            except Exception:
                pass  # can't parse pyproject — fall through to install

        if verbose:
            LOGGER.warning(f"Installing extra '{extra}'...")

        # From source checkout (editable install): install -e ".[extra]"
        # From PyPI install:                  install "boxmot[extra]"
        install_args = self._normalize_args(extra_args)
        if root_pyproject.is_file():
            # Editable install detected or running from source root
            # We use ROOT to point to the source directory
            target = f"{ROOT}[{extra}]"
            install_args.extend(["-e", target])
        else:
            # Installed from PyPI: install the published dist extra
            install_args.append(f"boxmot[{extra}]")

        self._run_install(install_args, description=f"extra '{extra}'", verbose=verbose)

    # ---------- internals ----------

    @staticmethod
    def _installer_command() -> list[str]:
        """Return the preferred package installer for the active environment."""
        return RequirementsChecker._installer_commands()[0]

    @staticmethod
    def _installer_commands() -> list[list[str]]:
        """Return installer command prefixes in fallback order."""
        commands: list[list[str]] = []
        if shutil.which("uv"):
            commands.append(["uv", "pip", "install", "--no-cache-dir"])
        commands.append([sys.executable, "-m", "pip", "install", "--no-cache-dir"])
        return commands

    @staticmethod
    def _normalize_args(args: Optional[Sequence[str] | str]) -> list[str]:
        """Return installer arguments from a sequence or shell-style string."""
        if args is None:
            return []
        if isinstance(args, str):
            return shlex.split(args)
        return [str(arg) for arg in args]

    def _missing_packages(self, requirements: Iterable[str]) -> list[str]:
        """Return requirement specifiers from *requirements* that are not satisfied."""
        return missing_requirements(requirements)

    def _install_packages(
        self, packages: Sequence[str], extra_args: Optional[Sequence[str] | str] = None
    ):
        """
        Install an explicit list of requirement specifiers.
        """
        LOGGER.warning(
            f"\nMissing or mismatched packages: {', '.join(packages)}\n"
            "Attempting installation..."
        )
        install_args = self._normalize_args(extra_args)
        install_args.extend(str(package) for package in packages)
        self._run_install(install_args, description="packages", verbose=True)

    def _run_install(
        self,
        install_args: Sequence[str],
        *,
        description: str,
        verbose: bool,
    ) -> None:
        """Run an install command, retrying with pip when uv fails."""
        failures: list[tuple[list[str], str, str]] = []
        commands = [[*installer, *install_args] for installer in self._installer_commands()]

        for idx, cmd in enumerate(commands):
            try:
                completed = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except FileNotFoundError as e:
                failures.append((cmd, "", str(e)))
            except subprocess.CalledProcessError as e:
                failures.append((cmd, e.stdout or "", e.stderr or str(e)))
            else:
                if verbose:
                    self._log_install_output(completed)
                    LOGGER.info(f"Installed {description} successfully.")
                return

            if idx + 1 < len(commands) and verbose:
                LOGGER.warning(
                    f"Installer command failed ({cmd[0]}). "
                    f"Retrying with {commands[idx + 1][0]}..."
                )

        if failures:
            last_cmd, stdout, stderr = failures[-1]
            tail = (stderr or stdout).strip().splitlines()[-5:]
            for line in tail:
                LOGGER.error(line)
            LOGGER.error(f"Failed to install {description} with: {' '.join(last_cmd)}")
        raise RuntimeError(f"Failed to install {description}")

    @staticmethod
    def _log_install_output(completed: subprocess.CompletedProcess[str]) -> None:
        """Log captured installer output through the project logger."""
        for line in (completed.stdout or "").splitlines():
            if line.strip():
                LOGGER.info(line)
        for line in (completed.stderr or "").splitlines():
            if line.strip():
                LOGGER.info(line)
