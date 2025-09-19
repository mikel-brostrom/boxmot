import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence

from boxmot.utils import logger as LOGGER
from packaging.requirements import Requirement
from importlib.metadata import version, PackageNotFoundError


REQUIREMENTS_FILE = Path("requirements.txt")


class RequirementsChecker:
    def __init__(self, group: str | None = None, requirements_file: Path = REQUIREMENTS_FILE):
        """
        If `group` is provided, we'll sync that uv dependency-group (or extra).
        Otherwise we'll read requirements_file and pip/uv-install missing packages.
        """
        self.group = group
        self.requirements_file = requirements_file
        self._uv = shutil.which("uv") is not None

    # ---------- public API ----------

    def check_packages(self, requirements: Iterable[str], extra_args: Optional[Sequence[str]] = None):
        """
        Check & install packages specified by requirement strings, e.g. ["foo", "bar>=1.2"].

        :param requirements: iterable of requirement specifiers as strings
        :param extra_args: extra args for the installer (e.g. ["--upgrade"])
        """
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
        if not self.requirements_file.is_file():
            LOGGER.warning(f"No requirements file found at {self.requirements_file.resolve()}")
            return

        reqs: list[str] = []
        for line in self.requirements_file.read_text().splitlines():
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
    ):
        """
        Sync a uv dependency-group OR install a project extra (PEP 621 optional-dependencies).

        :param group: name of the [tool.uv.group] to install (requires uv)
        :param extra: name of the [project.optional-dependencies] extra to install (uv or pip)
        :param extra_args: additional args passed to the installer
        """
        if bool(group) == bool(extra):  # both None or both set
            raise ValueError("Must provide exactly one of 'group' or 'extra'.")

        name = group or extra
        kind = "group" if group else "extra"
        LOGGER.warning(f"Installing {kind} '{name}'...")

        # Prefer uv if available. Groups require uv. Extras can be uv or pip.
        try:
            if group:
                if not self._uv:
                    raise RuntimeError("uv not found on PATH, cannot sync dependency group.")
                cmd = ["uv", "sync", "--no-default-groups", "--group", name]
                if extra_args:
                    cmd.extend(extra_args)
                subprocess.check_call(cmd)

            else:  # extra
                if self._uv:
                    cmd = ["uv", "pip", "install", "--no-cache-dir", ".[{}]".format(name)]
                else:
                    cmd = ["pip", "install", ".[{}]".format(name)]
                if extra_args:
                    cmd.extend(extra_args)
                subprocess.check_call(cmd)

            LOGGER.info(f"{kind.capitalize()} '{name}' installed successfully.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to install {kind} '{name}': {e}")
            raise RuntimeError(f"Failed to install {kind} '{name}': {e}")

    # ---------- internals ----------

    def _install_packages(self, packages: Sequence[str], extra_args: Optional[Sequence[str]] = None):
        try:
            LOGGER.warning(
                f"\nMissing or mismatched packages: {', '.join(packages)}\n"
                "Attempting installation..."
            )
            if self._uv:
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
