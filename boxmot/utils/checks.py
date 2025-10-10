import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

from importlib.metadata import version, PackageNotFoundError
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

    def __init__(self, group: Optional[str] = None, requirements_file: Path = REQUIREMENTS_FILE):
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
                if req.specifier and not req.specifier.contains(inst_ver, prereleases=True):
                    LOGGER.error(
                        f"{name!r} has version {inst_ver} which does not satisfy {req.specifier}."
                    )
                    missing.append(str(req))

        if missing:
            self._install_packages(missing, extra_args)

    def check_requirements_file(self, extra_args: Optional[Sequence[str]] = None):
        """
        Parse requirements.txt (or a custom path) and install what’s missing.
        Comments and blank lines are ignored.
        """
        path = self.requirements_file
        if not path.is_file():
            LOGGER.warning(f"No requirements file found at {path.resolve()}")
            return

        reqs: list[str] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.append(line)

        if reqs:
            self.check_packages(reqs, extra_args=extra_args)

    def sync_group_or_extra(
        self,
        group: Optional[str] = None,
        extra: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        cmds: Optional[Sequence[str]] = None,  # legacy alias
    ):
        """
        Install either:
          - a uv dependency *group* (requires uv), or
          - a project *extra* (PEP 621 optional-dependencies), via uv or pip.

        Exactly one of `group` or `extra` must be provided.
        """
        if extra_args is None and cmds is not None:
            extra_args = list(cmds)

        if bool(group) == bool(extra):  # both None or both set
            raise ValueError("Must provide exactly one of 'group' or 'extra'.")

        name = group or extra
        kind = "group" if group else "extra"
        LOGGER.warning(f"Installing {kind} '{name}'...")

        try:
            if group:
                if not self._uv_available:
                    raise RuntimeError("uv not found on PATH, cannot sync dependency group.")
                cmd = ["uv", "sync", "--no-default-groups", "--extra", name]
                if extra_args:
                    cmd.extend(extra_args)
                subprocess.check_call(cmd)

            else:  # extra
                if self._uv_available:
                    cmd = ["uv", "pip", "install", "--no-cache-dir", f".[{name}]"]
                else:
                    cmd = ["pip", "install", f".[{name}]"]
                if extra_args:
                    cmd.extend(extra_args)
                subprocess.check_call(cmd)

            LOGGER.info(f"{kind.capitalize()} '{name}' installed successfully.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to install {kind} '{name}': {e}")
            raise RuntimeError(f"Failed to install {kind} '{name}': {e}")

    # ---------- internals ----------

    def _install_packages(self, packages: Sequence[str], extra_args: Optional[Sequence[str]] = None):
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
