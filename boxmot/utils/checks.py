import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Optional, Sequence

from packaging.requirements import Requirement

# Replace this import with your logger, or use logging.getLogger(__name__)
from boxmot.utils import logger as LOGGER

REQUIREMENTS_FILE = Path("requirements.txt")


class RequirementsChecker:
    """
    Runtime dependency helper.

    Features:
      - Check/install a list of requirement specifiers (e.g., ["yolox", "onnx>=1.15"])
      - Read and install from a requirements.txt
      - Install a uv dependency *group* (requires uv)
      - Install a project *extra* (PEP 621 optional-dependencies) via uv or pip
      - Falls back to pip if uv is not available
      - Backward-compatible alias: `cmds` == `extra_args`
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
        self._uv_available = shutil.which("uv") is not None

    # ---------- public API ----------

    def check_packages(
        self,
        requirements: Iterable[str],
        extra_args: Optional[Sequence[str]] = None,
        cmds: Optional[Sequence[str]] = None,  # legacy alias
    ):
        """
        Check & install packages specified by requirement strings.

        :param requirements: iterable of requirement specifiers as strings
        :param extra_args: extra args for the installer (e.g. ["--upgrade"])
        :param cmds: legacy alias for extra_args
        """
        if extra_args is None and cmds is not None:
            extra_args = list(cmds)

        specs = [Requirement(r) for r in requirements]
        missing: list[str] = []

        for req in specs:
            name = req.name
            try:
                inst_ver = version(name)
            except PackageNotFoundError:
                LOGGER.error(f"Package {name!r} is not installed.")
                missing.append(str(req))
            else:
                if req.specifier and not req.specifier.contains(
                    inst_ver, prereleases=True
                ):
                    LOGGER.error(
                        f"{name!r} has version {inst_ver} which does not satisfy {req.specifier}."
                    )
                    missing.append(str(req))

        if missing:
            self._install_packages(missing, extra_args)

    def sync_extra(
        self,
        extra: str,
        extra_args: Optional[Sequence[str]] = None,
    ):
        """
        Install a project *extra* (PEP 621 optional-dependencies).
        - From source checkout + uv available: uv pip install ".[extra]"
        - From source checkout w/o uv:        python -m pip install ".[extra]"
        - From PyPI install:                  python -m pip install "boxmot[extra]"
        """
        if not extra:
            raise ValueError("Extra name must be provided (e.g. 'openvino', 'export').")
        LOGGER.warning(f"Installing extra '{extra}'...")

        in_source = Path("pyproject.toml").is_file()
        cmd: list[str]

        if in_source:
            if self._uv_available:
                cmd = ["uv", "pip", "install", "--no-cache-dir", f".[{extra}]"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", f".[{extra}]"]
        else:
            # Installed from PyPI: install the published dist extra
            cmd = [sys.executable, "-m", "pip", "install", f"boxmot[{extra}]"]

        if extra_args:
            cmd.extend(extra_args)

        try:
            subprocess.check_call(cmd)
            LOGGER.info(f"Extra '{extra}' installed successfully.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to install extra '{extra}': {e}")
            raise RuntimeError(f"Failed to install extra '{extra}': {e}")

    # ---------- internals ----------

    def _install_packages(
        self, packages: Sequence[str], extra_args: Optional[Sequence[str]] = None
    ):
        """
        Install an explicit list of requirement specifiers with uv (if present) or pip.
        """
        try:
            LOGGER.warning(
                f"\nMissing or mismatched packages: {', '.join(packages)}\n"
                "Attempting installation..."
            )
            if self._uv_available:
                cmd = ["uv", "pip", "install", "--no-cache-dir"]
            else:
                cmd = ["pip", "install"]

            if extra_args:
                cmd += list(extra_args)
            cmd += list(packages)

            subprocess.check_call(cmd)
            LOGGER.info("All missing packages were installed successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to install packages: {e}")
            raise RuntimeError(f"Failed to install packages: {e}")
