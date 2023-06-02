from pathlib import Path
import subprocess
import pkg_resources

from boxmot.utils import REQUIREMENTS
from boxmot.utils import logger


class TestRequirements():
    
    def check_requirements(self):
        requirements = pkg_resources.parse_requirements(REQUIREMENTS.open())
        self.check_packages(requirements)
        
    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        
        s = '' # missing packages
        for r in requirements:
            r = str(r)
            try:
                pkg_resources.require(r)
            except Exception as e:
                s += f'"{r}" '
        if s:
            logger.warning(f'\nMissing packages: {s}\nAtempting installation...')
            try:
                subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True, stderr=subprocess.STDOUT)
            except Exception as e:
                logger.error(e)
                exit()
            logger.success('All the missing packages were installed successfully')
