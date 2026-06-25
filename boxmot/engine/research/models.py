from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from boxmot.utils.rich.core.ui import print_text

from .constants import DEFAULT_PROPOSAL_MODEL, DEFAULT_PROPOSAL_MODEL_KWARGS, RESEARCH_METRICS


@dataclass(frozen=True, slots=True)
class RegressionPenalties:
    hota_penalty: float = 2
    idf1_penalty: float = 1.0
    mota_penalty: float = 1.0
    hota_tolerance: float = 0.0
    idf1_tolerance: float = 0.0
    mota_tolerance: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "hota_penalty",
            "idf1_penalty",
            "mota_penalty",
            "hota_tolerance",
            "idf1_tolerance",
            "mota_tolerance",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "hota_penalty": float(self.hota_penalty),
            "idf1_penalty": float(self.idf1_penalty),
            "mota_penalty": float(self.mota_penalty),
            "hota_tolerance": float(self.hota_tolerance),
            "idf1_tolerance": float(self.idf1_tolerance),
            "mota_tolerance": float(self.mota_tolerance),
        }


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    tracker: str
    benchmark: str
    source: Path | None = None
    detector: Path | None = None
    reid: Path | None = None
    editable_files: tuple[str, ...] | None = None
    extra_context_files: tuple[str, ...] = ()
    train_sequences: tuple[str, ...] | None = None
    val_sequences: tuple[str, ...] | None = None
    validation_split: float = 0.25
    proposal_model: str = DEFAULT_PROPOSAL_MODEL
    proposal_model_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PROPOSAL_MODEL_KWARGS))
    penalties: RegressionPenalties = field(default_factory=RegressionPenalties)
    max_metric_calls: int = 24
    reflection_minibatch_size: int = 2
    eval_timeout: float = 60.0
    project: Path = Path("runs")
    name: str = "research"
    progress_bar: bool = True
    keep_workspace: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ResearchConfig:
        benchmark = getattr(args, "benchmark", None) or getattr(args, "data", "")
        detector = None
        if getattr(args, "detector_explicit", False) and getattr(args, "detector", None):
            detector = Path(args.detector[0])

        reid = None
        if getattr(args, "reid_explicit", False) and getattr(args, "reid", None):
            reid = Path(args.reid[0])

        proposal_model_kwargs = dict(getattr(args, "proposal_model_kwargs", DEFAULT_PROPOSAL_MODEL_KWARGS) or {})
        if "reasoning_effort" not in proposal_model_kwargs:
            proposal_model_kwargs["reasoning_effort"] = DEFAULT_PROPOSAL_MODEL_KWARGS["reasoning_effort"]

        proposal_api_key = getattr(args, "proposal_api_key", None)
        if proposal_api_key:
            proposal_model_kwargs["api_key"] = str(proposal_api_key)

        proposal_api_key_env = getattr(args, "proposal_api_key_env", None)
        if proposal_api_key_env:
            proposal_model_kwargs["api_key_env"] = str(proposal_api_key_env)

        return cls(
            tracker=str(getattr(args, "tracker", "")),
            benchmark=str(benchmark),
            source=Path(getattr(args, "source")) if getattr(args, "source", None) else None,
            detector=detector,
            reid=reid,
            proposal_model=str(getattr(args, "proposal_model", DEFAULT_PROPOSAL_MODEL)),
            proposal_model_kwargs=proposal_model_kwargs,
            penalties=RegressionPenalties(
                hota_penalty=float(getattr(args, "hota_penalty", 0.0)),
                idf1_penalty=float(getattr(args, "idf1_penalty", 1.0)),
                mota_penalty=float(getattr(args, "mota_penalty", 1.0)),
                hota_tolerance=float(getattr(args, "hota_tolerance", 0.0)),
                idf1_tolerance=float(getattr(args, "idf1_tolerance", 0.0)),
                mota_tolerance=float(getattr(args, "mota_tolerance", 0.0)),
            ),
            max_metric_calls=int(getattr(args, "max_metric_calls", 24)),
            reflection_minibatch_size=int(getattr(args, "reflection_minibatch_size", 2)),
            eval_timeout=float(getattr(args, "eval_timeout", 900.0)),
            project=Path(getattr(args, "project", "runs")),
            name=str(getattr(args, "name", "research")),
            keep_workspace=bool(getattr(args, "keep_workspace", False)),
        )


@dataclass(slots=True)
class ResearchResult:
    tracker: str
    benchmark: str
    proposal_model: str
    run_dir: Path
    best_candidate_dir: Path
    editable_files: tuple[str, ...]
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]
    baseline_summary: dict[str, int | float]
    best_summary: dict[str, int | float]
    delta_summary: dict[str, float]
    workspace_dir: Path | None = None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        def _format_metrics(metrics: Mapping[str, int | float], *, signed: bool = False) -> str:
            parts = []
            for metric in RESEARCH_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    fmt = f"{float(value):+.3f}" if signed else f"{float(value):.3f}"
                    parts.append(f"{metric}={fmt}")
            return " ".join(parts) if parts else "n/a"

        lines = [
            "RESEARCH SUMMARY",
            f"Tracker: {self.tracker}",
            f"Benchmark: {self.benchmark}",
            f"Proposal model: {self.proposal_model}",
            f"Baseline: {_format_metrics(self.baseline_summary)}",
            f"Best: {_format_metrics(self.best_summary)}",
            f"Delta: {_format_metrics(self.delta_summary, signed=True)}",
            f"Best candidate dir: {self.best_candidate_dir}",
        ]
        if self.workspace_dir is not None:
            lines.append(f"Workspace: {self.workspace_dir}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print_text(self.render())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker,
            "benchmark": self.benchmark,
            "proposal_model": self.proposal_model,
            "run_dir": str(self.run_dir),
            "best_candidate_dir": str(self.best_candidate_dir),
            "editable_files": list(self.editable_files),
            "train_sequences": list(self.train_sequences),
            "val_sequences": list(self.val_sequences),
            "baseline_summary": self.baseline_summary,
            "best_summary": self.best_summary,
            "delta_summary": self.delta_summary,
            "workspace_dir": None if self.workspace_dir is None else str(self.workspace_dir),
        }
