"""Ray actor lifecycle for independent trials with retained replay workers."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from boxmot.engine.tuning.search_space import normalize_trial_config


def build_tracker_trainable(tune: Any, options: SimpleNamespace) -> type:
    """Bind serializable workflow options without constructing driver resources.

    Ray stays an optional dependency: the caller supplies its loaded Tune
    module. Each reused actor owns one objective and its spawn pool; resetting
    a trial changes only the search configuration. Trackers, pipelines and
    sequence iterators are constructed anew inside every replay task.
    """
    from boxmot.engine.tuning.tuner import TrackerObjective

    class TrackerTrainable(tune.Trainable):
        """Evaluate one complete tracker configuration per Ray trial."""

        workflow_options = options

        def setup(self, config: dict[str, Any]) -> None:
            self._objective = TrackerObjective(deepcopy(self.workflow_options))
            self._finished = False

        def step(self) -> dict[str, Any]:
            if self._finished:
                raise RuntimeError("Tracker trial already completed; reset its configuration before reuse.")
            try:
                result = self._objective(normalize_trial_config(self.config))
            except BaseException:
                self.cleanup()
                raise
            self._finished = True
            return {**result, "done": True}

        def reset_config(self, new_config: dict[str, Any]) -> bool:
            """Reset only trial state, retaining the objective's input workers."""
            self.config = deepcopy(new_config)
            self._finished = False
            return True

        def cleanup(self) -> None:
            """Close retained workers when Ray retires or interrupts this actor."""
            objective = getattr(self, "_objective", None)
            if objective is not None:
                objective.close()

    return TrackerTrainable
