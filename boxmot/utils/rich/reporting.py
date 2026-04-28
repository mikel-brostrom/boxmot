from __future__ import annotations

from typing import Any, ClassVar, Sequence

import boxmot.utils.rich.ui as ui


class RichWorkflowReporter:
    """Base helper for commands that render a Rich workflow panel."""

    title: ClassVar[str]
    steps: ClassVar[Sequence[tuple[str, ui.StepState]]] = ()
    stderr: ClassVar[bool] = True
    transient: ClassVar[bool] = False
    start_on_create: ClassVar[bool] = True
    prefer_alt_screen: ClassVar[bool] = False
    prefer_compact_layout: ClassVar[bool] = False

    def __init__(self, args: Any) -> None:
        self.args = args

    def fields(self) -> Sequence[tuple[str, object]]:
        return ()

    def create(self) -> ui.WorkflowProgress:
        workflow = ui.create_workflow_progress(
            self.title,
            self.fields(),
            steps=self.steps,
            stderr=self.stderr,
            transient=self.transient,
        )
        workflow.prefer_alt_screen = self.prefer_alt_screen
        workflow.prefer_compact_layout = self.prefer_compact_layout
        if self.start_on_create:
            workflow.start()
        return workflow


class WorkflowDetailCallback:
    """Callable adapter that routes progress text into one workflow step."""

    def __init__(self, workflow: ui.WorkflowProgress, step: str, *, render: bool = True) -> None:
        self.workflow = workflow
        self.step = step
        self.render = render

    def __call__(self, message: str) -> None:
        self.workflow.set_detail(self.step, message, render=self.render)


class SilentProgressReporter:
    """No-op progress reporter used when another UI owns the terminal."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_report(self, trials: Any, done: bool = False) -> bool:
        return False

    def report(self, trials: Any, done: bool, *sys_info: Any) -> None:
        return None


class RichWorkflowCallback:
    """Serializable callback base that keeps Rich workflow state driver-local."""

    detail_step: ClassVar[str | None] = None
    _workflow: ClassVar[ui.WorkflowProgress | None] = None

    @classmethod
    def set_workflow(cls, workflow: ui.WorkflowProgress | None) -> None:
        cls._workflow = workflow

    def set_workflow_detail(self, detail: str | None) -> None:
        workflow = type(self)._workflow
        if workflow is not None:
            workflow.set_detail(type(self).detail_step, detail)

    def setup(self, **info: Any) -> None:
        return None

    def on_step_begin(self, iteration: int, trials: list, **info: Any) -> None:
        return None

    def on_step_end(self, iteration: int, trials: list, **info: Any) -> None:
        return None

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_restore(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_save(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_result(self, iteration: int, trials: list, trial: Any, result: dict, **info: Any) -> None:
        return None

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_recover(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_checkpoint(self, iteration: int, trials: list, trial: Any, checkpoint: Any, **info: Any) -> None:
        return None

    def on_experiment_end(self, trials: list, **info: Any) -> None:
        return None

    def get_state(self) -> None:
        return None

    def set_state(self, state: dict[str, Any] | None) -> None:
        return None
