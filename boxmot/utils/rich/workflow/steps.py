"""Single source of truth for pipeline step labels and compositions.

Every pipeline mode (track, eval, generate, export, tune, research) builds
its step list from the same atomic labels.  The :func:`compose` helper turns
a sequence of labels into the ``(label, state)`` tuples that
:class:`~boxmot.utils.rich.core.ui.WorkflowProgress` expects (first step
*active*, the rest *todo*).

Adding a new mode is one line::

    MY_STEPS = compose(SETUP, MY_NEW_STEP)
"""

from __future__ import annotations

from boxmot.utils.rich.core.ui import StepState

# ── Atomic step labels ──────────────────────────────────────────────────

SETUP = "Set up"
GENERATE = "Generate detections and embeddings"
TUNE_KF = "Tune Kalman filter"
TRACK = "Run tracker"
POSTPROCESS = "Postprocess tracks"
EVALUATE = "Evaluate results"
EXPORT = "Export to formats"
OPTIMIZE = "Optimize trials"

# Research-specific (no shared overlap with other modes)
PREPARE = "Prepare workspace"
BASELINE = "Baseline evaluation"
RESEARCH_OPTIMIZE = "GEPA optimization"
BEST_CANDIDATE = "Best candidate evaluation"


# ── Composition helper ──────────────────────────────────────────────────


def compose(*labels: str) -> tuple[tuple[str, StepState], ...]:
    """Build a pipeline step list: first step *active*, rest *todo*."""
    return tuple((label, "active" if i == 0 else "todo") for i, label in enumerate(labels))


# ── Pre-built pipeline compositions ─────────────────────────────────────

TRACK_STEPS = compose(SETUP, TRACK)
GENERATE_STEPS = compose(SETUP, GENERATE)
EXPORT_STEPS = compose(SETUP, EXPORT)
TUNE_STEPS = compose(SETUP, GENERATE, OPTIMIZE)
RESEARCH_STEPS = compose(PREPARE, BASELINE, RESEARCH_OPTIMIZE, BEST_CANDIDATE)


# ── Dynamic pipeline builders (canonical way to get mode-specific steps) ──


def eval_steps(*, tune_kf: bool = False, postprocess: bool = False) -> tuple[tuple[str, StepState], ...]:
    """Build eval pipeline steps, including optional stages only when enabled."""
    labels = [SETUP, GENERATE]
    if tune_kf:
        labels.append(TUNE_KF)
    labels.append(TRACK)
    if postprocess:
        labels.append(POSTPROCESS)
    labels.append(EVALUATE)
    return compose(*labels)


def tune_steps(*, tune_kf: bool = False) -> tuple[tuple[str, StepState], ...]:
    """Build tune pipeline steps, including optional KF tuning stage."""
    labels = [SETUP, GENERATE]
    if tune_kf:
        labels.append(TUNE_KF)
    labels.append(OPTIMIZE)
    return compose(*labels)
