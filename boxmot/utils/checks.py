import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional, Sequence

from packaging.requirements import Requirement

# Replace this import with your logger, or use logging.getLogger(__name__)
from boxmot.utils import logger as LOGGER, ROOT

REQUIREMENTS_FILE = Path("requirements.txt")


class RequirementsChecker:
    """
    Runtime dependency helper.

    Features:
      - Check/install a list of requirement specifiers (e.g., ["yolox", "onnx>=1.15"])
      - Read and install from a requirements.txt
      - Install a uv dependency *group* (requires uv)
      - Install a project *extra* (PEP 621 optional-dependencies) via uv
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
        extra_args: Optional[Sequence[str]] = None,
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
        extra_args: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ):
        """
        Install a project *extra* (PEP 621 optional-dependencies).
        - From source checkout + uv available: uv pip install -e ".[extra]"
        - From PyPI install:                  uv pip install "boxmot[extra]"
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

        cmd: list[str]

        # From source checkout (editable install): uv pip install -e ".[extra]"
        # From PyPI install: uv pip install "boxmot[extra]"
        if root_pyproject.is_file():
            # Editable install detected or running from source root
            # We use ROOT to point to the source directory
            target = f"{ROOT}[{extra}]"
            cmd = ["uv", "pip", "install", "--no-cache-dir", "-e", target]
        else:
            # Installed from PyPI: install the published dist extra
            cmd = ["uv", "pip", "install", f"boxmot[{extra}]"]

        if extra_args:
            cmd.extend(extra_args)

        try:
            # Always pipe subprocess output so that callers running inside a
            # Rich ``Live`` workflow do not see raw ``uv`` writes corrupting
            # their cursor positioning. When verbose is requested we surface
            # the captured output via the LOGGER (which routes through Rich).
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                for line in (completed.stdout or "").splitlines():
                    if line.strip():
                        LOGGER.info(line)
                for line in (completed.stderr or "").splitlines():
                    if line.strip():
                        LOGGER.info(line)
                LOGGER.info(f"Extra '{extra}' installed successfully.")
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "").strip().splitlines()[-5:]
            for line in stderr_tail:
                LOGGER.error(line)
            LOGGER.error(f"Failed to install extra '{extra}': {e}")
            raise RuntimeError(f"Failed to install extra '{extra}': {e}")

    # ---------- internals ----------

    def _missing_packages(self, requirements: Iterable[str]) -> list[str]:
        """Return requirement specifiers from *requirements* that are not satisfied."""
        missing: list[str] = []
        for req in [Requirement(r) for r in requirements]:
            if req.marker is not None and not req.marker.evaluate():
                continue
            try:
                inst_ver = version(req.name)
                if req.specifier and not req.specifier.contains(inst_ver, prereleases=True):
                    missing.append(str(req))
            except PackageNotFoundError:
                missing.append(str(req))
        return missing

    def _install_packages(
        self, packages: Sequence[str], extra_args: Optional[Sequence[str]] = None
    ):
        """
        Install an explicit list of requirement specifiers with uv.
        """
        try:
            LOGGER.warning(
                f"\nMissing or mismatched packages: {', '.join(packages)}\n"
                "Attempting installation..."
            )
            cmd = ["uv", "pip", "install", "--no-cache-dir"]

            if extra_args:
                cmd += list(extra_args)
            cmd += list(packages)

            # Pipe subprocess output (see comment in ``sync_extra``) and
            # surface any captured lines through the LOGGER so they render
            # cleanly inside an active Rich workflow.
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in (completed.stdout or "").splitlines():
                if line.strip():
                    LOGGER.info(line)
            for line in (completed.stderr or "").splitlines():
                if line.strip():
                    LOGGER.info(line)
            LOGGER.info("All missing packages were installed successfully.")
        except subprocess.CalledProcessError as e:
            stderr_tail = (e.stderr or "").strip().splitlines()[-5:]
            for line in stderr_tail:
                LOGGER.error(line)
            LOGGER.error(f"Failed to install packages: {e}")
            raise RuntimeError(f"Failed to install packages: {e}")
        except Exception as e:
            LOGGER.error(f"Failed to install packages: {e}")
            raise RuntimeError(f"Failed to install packages: {e}")
