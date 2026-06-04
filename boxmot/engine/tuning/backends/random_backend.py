"""Random search backend (no Bayesian optimizer)."""
from __future__ import annotations

from typing import Any

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.search_space import yaml_to_tune_space


class RandomBackend(BaseTuneBackend):
    """Random search — uses Ray Tune's native param_space with no searcher."""

    def build(self, tune) -> tuple[None, dict[str, Any]]:
        return None, yaml_to_tune_space(self.yaml_cfg, tune)
