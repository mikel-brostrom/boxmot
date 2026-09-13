"""Random search backend (no Bayesian optimizer)."""

from __future__ import annotations

from typing import Any

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.search_space import yaml_to_tune_space


class RandomBackend(BaseTuneBackend):
    """Random search using Ray Tune's native parameter-space sampler."""

    def build(self, tune: Any) -> tuple[Any | None, dict[str, Any]]:
        """Use an explicitly seeded sampler when a reproducible run is requested."""
        param_space = yaml_to_tune_space(self.yaml_cfg, tune)
        if self.seed is None:
            return None, param_space

        from ray.tune.search.basic_variant import BasicVariantGenerator

        return BasicVariantGenerator(random_state=self.seed), param_space
