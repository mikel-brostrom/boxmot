"""Abstract base class for tune search backends."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any


class BaseTuneBackend(ABC):
    """Interface that all tune search backends must implement.

    Subclasses are responsible for constructing the Ray Tune search algorithm
    and the ``param_space`` dict passed to ``tune.Tuner``.
    """

    def __init__(
        self,
        *,
        yaml_cfg: dict,
        opt_metrics: list[str],
        opt_modes: list[str],
        baseline_config: dict[str, Any] | None = None,
        seed: int | None = None,
        max_concurrent: int = 4,
    ):
        self.yaml_cfg = yaml_cfg
        self.opt_metrics = opt_metrics
        self.opt_modes = opt_modes
        self.baseline_config = baseline_config
        self.seed = seed
        self.max_concurrent = max_concurrent

    @abstractmethod
    def build(self, tune) -> tuple[Any | None, dict[str, Any]]:
        """Return ``(search_alg, param_space)`` for Ray Tune.

        Parameters
        ----------
        tune : module
            The ``ray.tune`` module (passed to avoid import at class definition time).

        Returns
        -------
        search_alg : ConcurrencyLimiter | None
            Wrapped searcher, or None for random search.
        param_space : dict
            Flat dict of Ray Tune samplers (empty when the searcher owns the space).
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs that are not accepted by the installed Ray searcher version."""
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs

        parameters = signature.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in parameters}
