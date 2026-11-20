# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
# Full license text: https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE

import logging
import os
from pathlib import Path

from boxmot.resources.paths import default_weights_directory, repository_root

ROOT = repository_root() or Path(__file__).resolve().parents[2]

TOML = ROOT / "pyproject.toml"

BOXMOT     = ROOT / "boxmot"
ENGINE     = BOXMOT / "engine"
DATASETS = ROOT / "datasets"
MOT_DATASETS = DATASETS / "mot"
REID_DATASETS = DATASETS / "reid"

WEIGHTS = default_weights_directory()
DATA = ROOT / "data"
BENCHMARK_DATA = MOT_DATASETS

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads


# Library imports leave handlers, levels, and propagation to the application.
# Command entry points opt into Rich output through engine.ui.logging.
logger = logging.getLogger("boxmot")
