# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import pkg_resources
import logging
from pathlib import Path
import pip

logger = logging.getLogger(__name__)
REQUIREMENTS = Path('requirements.txt')

class RequirementsChecker:

    def check_requirements(self):
        requirements = pkg_resources.parse_requirements(REQUIREMENTS.open())
        self.check_packages(requirements)

    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        missing_packages = []
        for r in requirements:
            try:
                pkg_resources.require(str(r))
            except Exception as e:
                logger.error(f'{e}')
                missing_packages.append(str(r))
        
        if missing_packages:
            self.install_packages(missing_packages, cmds)

    def install_packages(self, packages, cmds=''):
        try:
            logger.warning(f'\nMissing packages: {", ".join(packages)}\nAttempting installation...')
            pip_args = ['install', '--no-cache-dir'] + packages + cmds.split()
            pip.main(pip_args)
            logger.info('All the missing packages were installed successfully')
        except Exception as e:
            logger.error(f'Failed to install packages: {e}')
            exit()