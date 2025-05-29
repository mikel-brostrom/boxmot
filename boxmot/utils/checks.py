import subprocess
from pathlib import Path
from typing import Optional

import pkg_resources

from boxmot.utils import logger as LOGGER

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
        # parse requirements.txt
        with self.requirements_file.open() as f:
            reqs = pkg_resources.parse_requirements(f)
        self._check_packages(reqs)

    def check_packages(self, requirements, cmds=[]):
        missing = []
        for r in requirements:
            try:
                pkg_resources.require(str(r))
            except Exception as e:
                LOGGER.error(f"{e}")
                missing.append(str(r))

        if missing:
            self.install_packages(missing, cmds)

    def install_packages(self, packages, cmds=[]):
        try:
            LOGGER.warning(
                f"\nMissing packages: {', '.join(packages)}\nAttempting installation..."
            )
            pip_cmd = ["uv", "pip", "install", "--no-cache-dir"] + cmds + packages
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
