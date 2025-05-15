# requirements_checker.py

import pkg_resources
import subprocess
from pathlib import Path
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

    def _check_packages(self, requirements):
        missing = []
        for r in requirements:
            try:
                pkg_resources.require(str(r))
            except Exception as e:
                LOGGER.error(f"{e}")
                missing.append(str(r))

        if missing:
            self._install_packages(missing)

    def _install_packages(self, packages):
        try:
            LOGGER.warning(
                f"\nMissing packages: {', '.join(packages)}\nAttempting installation..."
            )
            pip_cmd = ["uv", "pip", "install", "--no-cache-dir"] + packages
            subprocess.check_call(pip_cmd)
            LOGGER.info("All the missing packages were installed successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to install packages: {e}")
            raise RuntimeError(f"Failed to install packages: {e}")

    def _sync_group(self, group: str):
        try:
            LOGGER.warning(f"Syncing dependency-group '{group}'...")
            sync_cmd = [
                "uv", "sync",
                "--no-default-groups",
                "--group", group
            ]
            subprocess.check_call(sync_cmd)
            LOGGER.info(f"Dependency-group '{group}' installed successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to sync group '{group}': {e}")
            raise RuntimeError(f"Failed to sync group '{group}': {e}")


if __name__ == "__main__":
    # Example usages:
    # 1) to install a tflite group:
    RequirementsChecker(group="tflite").check_requirements()

    # 2) to fall back on a requirements.txt:
    # RequirementsChecker().check_requirements()
