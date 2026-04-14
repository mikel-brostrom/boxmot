from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from boxmot.engine.results import Results, Tracks
from boxmot.utils.compat import dataclass_slots_kwargs

from . import _reporting as reporting


def _results_summary_snapshot(results: Results, source: Any) -> dict[str, Any]:
    frames = int(results.totals["frames"])
    avg_total = (results.totals["total"] / frames) if frames else 0.0
    return {
        "source": str(source),
        "frames": frames,
        "detections": int(results.totals["detections"]),
        "tracks": int(results.totals["tracks"]),
        "unique_tracks": len(getattr(results, "_track_ids_seen", set())),
        "timings_ms": {
            "det": float(results.totals["det"]),
            "reid": float(results.totals["reid"]),
            "track": float(results.totals["track"]),
            "total": float(results.totals["total"]),
            "avg_total": float(avg_total),
        },
    }


def _track_timings_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    timings = dict(summary.get("timings_ms", {}))
    avg_total = float(timings.get("avg_total", 0.0) or 0.0)
    timings["fps"] = (1000.0 / avg_total) if avg_total else 0.0
    return timings


@dataclass(**dataclass_slots_kwargs())
class ValidationResult:
    benchmark: str
    raw: dict[str, Any]
    summary_label: str
    summary: dict[str, Any]
    exp_dir: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    args: Any = None

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"ValidationResult(benchmark={self.benchmark!r}, "
            f"summary={self.summary!r}, exp_dir={self.exp_dir!r})"
        )

    def render(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> str:
        return reporting.render_validation_cli_report(
            self.raw,
            args=self.args,
            timings=self.timings,
            title=reporting.CLI_RESULTS_SUMMARY_TITLE if title is None else title,
            include_sequences=include_sequences,
            include_timings=include_timings,
        )

    def format_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        report_title = reporting.DEFAULT_VALIDATION_REPORT_TITLE if title is None else title
        return reporting.format_validation_report(
            self.raw,
            args=self.args,
            title=report_title,
            include_sequences=include_sequences,
        )

    def print_report(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
    ) -> None:
        print(
            self.render(
                title=title,
                include_sequences=include_sequences,
                include_timings=include_timings,
            )
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload = {
            "benchmark": self.benchmark,
            "summary_label": self.summary_label,
            "summary": dict(self.summary),
            "timings": dict(self.timings),
            "exp_dir": None if self.exp_dir is None else str(self.exp_dir),
        }
        if include_raw:
            payload["raw"] = self.raw
        return payload


@dataclass(**dataclass_slots_kwargs())
class TuneTrialResult:
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
        print(
            self.render(
                title=title,
                include_sequences=include_sequences,
                include_timings=include_timings,
            )
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        return {
            "index": self.index,
            "config": dict(self.config),
            "score": list(self.score),
            "metrics": self.metrics.to_dict(include_raw=include_raw),
        }


@dataclass(**dataclass_slots_kwargs())
class TuneResult:
    benchmark: str
    tracker: str
    trials: list[TuneTrialResult]
    best: TuneTrialResult
    best_config: dict[str, Any]
    best_yaml: Path

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
        return reporting.render_validation_cli_report(
            self.best.raw,
            args=self.best.args,
            timings=self.best.timings,
            title=reporting.CLI_TUNE_BEST_SUMMARY_TITLE if title is None else title,
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
        print(
            self.render(
                title=title,
                include_sequences=include_sequences,
                include_timings=include_timings,
            )
        )

    def format_best_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        report_title = reporting.DEFAULT_TUNE_BEST_REPORT_TITLE if title is None else title
        return reporting.format_validation_report(
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
        print(
            self.render(
                title=title,
                include_sequences=include_sequences,
                include_timings=include_timings,
            )
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


@dataclass(**dataclass_slots_kwargs())
class ExportResult:
    weights: Path
    files: dict[str, Any]


@dataclass(**dataclass_slots_kwargs())
class TrackRunResult:
    source: Any
    results: Results
    video_path: Path | None
    text_path: Path | None
    _timings: dict[str, Any] = field(default_factory=dict, repr=False)
    _summary: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def timings(self) -> dict[str, Any]:
        self.refresh()
        return self._timings

    @property
    def summary(self) -> dict[str, Any]:
        self.refresh()
        return self._summary

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        self.refresh()
        return (
            f"TrackRunResult(source={self.source!r}, summary={self._summary!r}, "
            f"video_path={self.video_path!r}, text_path={self.text_path!r})"
        )

    def __iter__(self) -> Iterator[Tracks]:
        for track_result in self.results:
            self.refresh()
            yield track_result
        self.refresh()

    def show(self) -> None:
        self.results.show()
        self.refresh()

    def stop(self, reason: str | None = None) -> None:
        self.results.stop(reason)
        self.refresh()

    def format_summary(self) -> str:
        self.refresh()
        return self.results.format_summary()

    def render(self) -> str:
        return self.format_summary()

    def print_summary(self) -> None:
        print(self.render())

    def refresh(self) -> None:
        summary_fn = getattr(self.results, "summary", None)
        if callable(summary_fn):
            self._summary = summary_fn()
        else:
            self._summary = _results_summary_snapshot(self.results, self.source)
        self._timings = _track_timings_from_summary(self._summary)

