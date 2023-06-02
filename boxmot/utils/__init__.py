import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
BOXMOT = ROOT / 'boxmot'
EXAMPLES = ROOT / 'examples'
WEIGHTS = ROOT / 'examples' / 'weights'
REQUIREMENTS = ROOT / 'requirements.txt'

# global logger
from loguru import logger
logger.remove()
logger.add(sys.stderr, colorize=True)