"""Structured results returned by tracker tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boxmot.engine.eval.results import ValidationResult


@dataclass(slots=True)
class TuneTrialResult:
    """Metrics and configuration captured for one tuning trial."""

    index: int
    config: dict[str, Any]
    metrics: ValidationResult
    score: tuple[float, ...]

    @property
    def benchmark(self) -> str:
        return self.metrics.benchmark

    @property
    def raw(self) -> dict[str, Any]:
        return self.metrics.raw

    @property
    def summary_label(self) -> str:
        return self.metrics.summary_label

    @property
    def summary(self) -> dict[str, Any]:
        return self.metrics.summary

    @property
    def timings(self) -> dict[str, Any]:
        return self.metrics.timings

    @property
    def exp_dir(self) -> Path | None:
        return self.metrics.exp_dir

    @property
    def args(self) -> Any:
        return self.metrics.args

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"TuneTrialResult(index={self.index}, summary={self.summary!r}, "
            f"config={self.config!r}, exp_dir={self.exp_dir!r})"
        )

    def render(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> str:
        return self.metrics.render(
            title=title,
            include_sequences=include_sequences,
            include_timings=include_timings,
        )

    def format_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        return self.metrics.format_report(title=title, include_sequences=include_sequences)

    def print_report(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> None:
        self.metrics.print_report(
            title=title,
            include_sequences=include_sequences,
            include_timings=include_timings,
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        return {
            "index": self.index,
            "config": dict(self.config),
            "score": list(self.score),
            "metrics": self.metrics.to_dict(include_raw=include_raw),
        }


@dataclass(slots=True)
class TuneResult:
    """Structured result for a completed tracker tuning workflow."""

    benchmark: str
    tracker: str
    trials: list[TuneTrialResult]
    best: TuneTrialResult
    best_config: dict[str, Any]
    best_yaml: Path
    workflow_rendered: bool = False

    @property
    def summary_label(self) -> str:
        return self.best.summary_label

    @property
    def summary(self) -> dict[str, Any]:
        return self.best.summary

    @property
    def raw(self) -> dict[str, Any]:
        return self.best.raw

    @property
    def timings(self) -> dict[str, Any]:
        return self.best.timings

    @property
    def exp_dir(self) -> Path | None:
        return self.best.exp_dir

    @property
    def args(self) -> Any:
        return self.best.args

    @property
    def baseline(self) -> TuneTrialResult:
        return self.trials[0]

    def __str__(self) -> str:
        if self.workflow_rendered:
            return ""
        return self.render()

    def __repr__(self) -> str:
        return (
            f"TuneResult(benchmark={self.benchmark!r}, tracker={self.tracker!r}, "
            f"summary={self.summary!r}, best_config={self.best_config!r}, best_yaml={self.best_yaml!r})"
        )

    def render(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> str:
        from boxmot.engine.ui.reporters.validation import (
            CLI_TUNE_BEST_SUMMARY_TITLE,
            render_validation_cli_report,
        )

        return render_validation_cli_report(
            self.best.raw,
            args=self.best.args,
            timings=self.best.timings,
            title=CLI_TUNE_BEST_SUMMARY_TITLE if title is None else title,
            include_sequences=include_sequences,
            include_timings=include_timings,
            compare_raw=self.baseline.raw,
            compare_args=self.baseline.args,
        )

    def format_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        return self.format_best_report(title=title, include_sequences=include_sequences)

    def print_report(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> None:
        from boxmot.engine.ui.reporters.validation import (
            CLI_TUNE_BEST_SUMMARY_TITLE,
            print_validation_cli_report,
        )

        print_validation_cli_report(
            self.best.raw,
            args=self.best.args,
            timings=self.best.timings,
            title=CLI_TUNE_BEST_SUMMARY_TITLE if title is None else title,
            include_sequences=include_sequences,
            include_timings=include_timings,
            compare_raw=self.baseline.raw,
            compare_args=self.baseline.args,
        )

    def format_best_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        from boxmot.engine.ui.reporters.validation import (
            DEFAULT_TUNE_BEST_REPORT_TITLE,
            format_validation_report,
        )

        report_title = DEFAULT_TUNE_BEST_REPORT_TITLE if title is None else title
        return format_validation_report(
            self.best.raw,
            args=self.best.args,
            title=report_title,
            include_sequences=include_sequences,
        )

    def print_best_report(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> None:
        self.print_report(
            title=title,
            include_sequences=include_sequences,
            include_timings=include_timings,
        )

    def to_dict(self, *, include_trials: bool = False, include_raw: bool = False) -> dict[str, Any]:
        payload = {
            "benchmark": self.benchmark,
            "tracker": self.tracker,
            "summary_label": self.summary_label,
            "summary": dict(self.summary),
            "best_config": dict(self.best_config),
            "best_yaml": str(self.best_yaml),
            "best": self.best.to_dict(include_raw=include_raw),
        }
        if include_trials:
            payload["trials"] = [trial.to_dict(include_raw=include_raw) for trial in self.trials]
        return payload


__all__ = ("TuneResult", "TuneTrialResult")
