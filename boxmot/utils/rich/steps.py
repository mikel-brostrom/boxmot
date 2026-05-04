"""Single source of truth for pipeline step labels and compositions.

Every pipeline mode (track, eval, generate, export, tune, research) builds
its step list from the same atomic labels.  The :func:`compose` helper turns
a sequence of labels into the ``(label, state)`` tuples that
:class:`~boxmot.utils.rich.ui.WorkflowProgress` expects (first step
*active*, the rest *todo*).

Adding a new mode is one line::

    MY_STEPS = compose(SETUP, MY_NEW_STEP)
"""

from __future__ import annotations

from boxmot.utils.rich.ui import StepState

# ── Atomic step labels ──────────────────────────────────────────────────

SETUP    = "Set up"
GENERATE = "Generate detections and embeddings"
TRACK    = "Run tracker"
EVALUATE = "Evaluate results"
EXPORT   = "Export to formats"
OPTIMIZE = "Optimize trials"

# Research-specific (no shared overlap with other modes)
PREPARE            = "Prepare workspace"
BASELINE           = "Baseline evaluation"
RESEARCH_OPTIMIZE  = "GEPA optimization"
BEST_CANDIDATE     = "Best candidate evaluation"


# ── Composition helper ──────────────────────────────────────────────────

def compose(*labels: str) -> tuple[tuple[str, StepState], ...]:
    """Build a pipeline step list: first step *active*, rest *todo*."""
    return tuple(
        (label, "active" if i == 0 else "todo")
        for i, label in enumerate(labels)
    )


# ── Pre-built pipeline compositions ─────────────────────────────────────

TRACK_STEPS    = compose(SETUP, TRACK)
GENERATE_STEPS = compose(SETUP, GENERATE)
EVAL_STEPS     = compose(SETUP, GENERATE, TRACK, EVALUATE)
EXPORT_STEPS   = compose(SETUP, EXPORT)
TUNE_STEPS     = compose(SETUP, GENERATE, OPTIMIZE)
RESEARCH_STEPS = compose(PREPARE, BASELINE, RESEARCH_OPTIMIZE, BEST_CANDIDATE)
