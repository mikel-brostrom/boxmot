import subprocess
from pathlib import Path
from typing import Iterable, Optional

from boxmot.utils import logger as LOGGER
from packaging.requirements import Requirement
from importlib.metadata import version, PackageNotFoundError


REQUIREMENTS_FILE = Path("requirements.txt")


class RequirementsChecker:
    def __init__(self, group: str = None, requirements_file: Path = REQUIREMENTS_FILE):
        """
        If `group` is provided, we'll sync that PDM/uv dependency-group.
        Otherwise we'll read requirements_file and pip-install missing packages.
        """
        self.group = group
        self.requirements_file = requirements_file

    def check_requirements(self):
        if self.group:
            self._sync_group(self.group)
        else:
            self._check_from_requirements()

    def _check_from_requirements(self):
        # parse requirements.txt into Requirement objects
        reqs = []
        with self.requirements_file.open() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                reqs.append(Requirement(line))
        self._check_packages(reqs)

    def check_packages(self, requirements: Iterable[str], cmds: Optional[list[str]] = None):
        """
        Check and install packages specified by requirement strings, e.g.
        ["foo", "bar>=1.2"].

        :param requirements: iterable of requirement specifiers as strings
        :param cmds: extra pip args (e.g. ["--upgrade"]).
        """
        # turn each string into a Requirement
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
            self.install_packages(missing, cmds)

    def install_packages(self, packages, extra_pip_args=None):
        try:
            LOGGER.warning(
                f"\nMissing or mismatched packages: {', '.join(packages)}\n"
                "Attempting installation..."
            )
            pip_cmd = ["uv", "pip", "install", "--no-cache-dir"] + (extra_pip_args or []) + packages
            subprocess.check_call(pip_cmd)
            LOGGER.info("All the missing packages were installed successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to install packages: {e}")
            raise RuntimeError(f"Failed to install packages: {e}")

    def sync_group_or_extra(
        self, group: Optional[str] = None, extra: Optional[str] = None
    ):
        """
        Sync a uv dependency-group or an extra.

        :param group: name of the [tool.uv.group] to install
        :param extra: name of the [project.optional-dependencies] extra to install
        """
        if bool(group) == bool(extra):
            # both None or both set
            raise ValueError("Must provide exactly one of 'group' or 'extra'.")

        name = group or extra
        kind = "group" if group else "extra"
        LOGGER.warning(f"Syncing {kind} '{name}'...")

        cmd = ["uv", "sync", "--no-default-groups", f"--{kind}", name]

        try:
            subprocess.check_call(cmd)
            LOGGER.info(f"{kind.capitalize()} '{name}' installed successfully.")
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Failed to sync {kind} '{name}': {e}")
            raise RuntimeError(f"Failed to sync {kind} '{name}': {e}")


if __name__ == "__main__":
    # Example usages:
    # 1) to install a tflite group:
    RequirementsChecker(group="tflite").check_requirements()

    # 2) to fall back on a requirements.txt:
    # RequirementsChecker().check_requirements()
