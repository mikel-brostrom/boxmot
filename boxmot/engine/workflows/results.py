from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from boxmot.utils.rich.core.ui import (
    STYLE_ACCENT,
    STYLE_RULE,
    STYLE_TABLE_HEADER,
    STYLE_TEXT,
    STYLE_TEXT_STRONG,
    print_text,
)

from . import reporting

if TYPE_CHECKING:
    from boxmot.engine.tracking.results import FrameResult, Results


def _results_summary_snapshot(results: Results, source: Any) -> dict[str, Any]:
    frames = int(results.totals["frames"])
    avg_total = (results.totals["total"] / frames) if frames else 0.0
    return {
        "source": str(source),
        "frames": frames,
        "detections": int(results.totals["detections"]),
        "tracks": int(results.totals["tracks"]),
        "unique_tracks": len(getattr(results, "_track_ids_seen", set())),
        "timing_metadata": dict(getattr(results, "timing_metadata", {})),
        "timings_ms": {
            "det": float(results.totals["det"]),
            "detector_preprocess": float(results.totals.get("detector_preprocess", 0.0)),
            "detector_process": float(results.totals.get("detector_process", 0.0)),
            "detector_postprocess": float(results.totals.get("detector_postprocess", 0.0)),
            "reid": float(results.totals["reid"]),
            "reid_preprocess": float(results.totals.get("reid_preprocess", 0.0)),
            "reid_process": float(results.totals.get("reid_process", 0.0)),
            "reid_postprocess": float(results.totals.get("reid_postprocess", 0.0)),
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


def _timing_fps(total_ms: float, frames: int) -> float:
    avg_ms = (float(total_ms) / frames) if frames else 0.0
    return (1000.0 / avg_ms) if avg_ms else 0.0


def _build_tracking_summary_stats_table(summary: dict[str, Any]) -> Table:
    table = Table.grid(expand=False, padding=(0, 1))
    table.add_column(style=STYLE_ACCENT, no_wrap=True)
    table.add_column(style=STYLE_TEXT)

    rows = (
        ("Frames", int(summary.get("frames", 0) or 0)),
        ("Detections", int(summary.get("detections", 0) or 0)),
        ("Track rows", int(summary.get("tracks", 0) or 0)),
        ("Unique IDs", int(summary.get("unique_tracks", 0) or 0)),
    )
    for label, value in rows:
        table.add_row(Text(label, style=STYLE_ACCENT), Text(str(value), style=STYLE_TEXT))
    return table


def _build_tracking_summary_timing_table(summary: dict[str, Any]) -> Table:
    timings = dict(summary.get("timings_ms", {}))
    frames = int(summary.get("frames", 0) or 0)
    breakdown = reporting.derive_timing_breakdown(timings, frames, total_time_ms=timings.get("total"))
    metadata = summary.get("timing_metadata") if isinstance(summary.get("timing_metadata"), dict) else {}

    table = Table(
        expand=True,
        box=None,
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        row_styles=["", ""],
        pad_edge=False,
        show_edge=False,
        padding=(0, 2),
        collapse_padding=False,
    )
    table.add_column("Stage", style=STYLE_TEXT_STRONG, no_wrap=True, ratio=3)
    table.add_column("Total (ms)", justify="right", no_wrap=True, ratio=1)
    table.add_column("Avg (ms)", justify="right", no_wrap=True, ratio=1)
    table.add_column("FPS", justify="right", no_wrap=True, ratio=1)

    for entry in reporting.build_timing_display_rows(
        breakdown,
        frames,
        metadata=metadata,
        overall_avg_ms=float(timings.get("avg_total", 0.0) or 0.0),
    ):
        if entry["kind"] == "group":
            table.add_row(Text(str(entry["label"]), style=STYLE_ACCENT), "", "", "")
            continue
        if entry["kind"] == "note":
            table.add_row(Text(str(entry["label"]), style=STYLE_TEXT), "", "", "")
            continue

        row_style = STYLE_TEXT_STRONG if bool(entry["strong"]) else None
        table.add_row(
            str(entry["label"]),
            f"{float(entry['total']):.1f}",
            f"{float(entry['avg']):.2f}",
            f"{float(entry['fps']):.1f}",
            style=row_style,
        )

    return table


def _build_tracking_summary_renderable(summary: dict[str, Any]) -> RenderableType:
    return Group(
        Rule("TRACKING SUMMARY", style=STYLE_RULE),
        _build_tracking_summary_stats_table(summary),
        Rule(style=STYLE_RULE),
        _build_tracking_summary_timing_table(summary),
    )


@dataclass(slots=True)
class ValidationResult:
    benchmark: str
    raw: dict[str, Any]
    summary_label: str
    summary: dict[str, Any]
    exp_dir: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    args: Any = None
    workflow_rendered: bool = False

    def __str__(self) -> str:
        if self.workflow_rendered:
            return ""
        return self.render()

    def __repr__(self) -> str:
        return f"ValidationResult(benchmark={self.benchmark!r}, summary={self.summary!r}, exp_dir={self.exp_dir!r})"

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

    def renderable(
        self,
        *,
        title: str | None = None,
        include_sequences: bool = True,
        include_timings: bool = False,
        compare_raw: dict[str, Any] | None = None,
        compare_args: Any = None,
    ) -> RenderableType:
        return reporting.build_validation_cli_renderable(
            self.raw,
            args=self.args,
            timings=self.timings,
            title=title,
            include_sequences=include_sequences,
            include_timings=include_timings,
            compare_raw=compare_raw,
            compare_args=compare_args,
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
        reporting.print_validation_cli_report(
            self.raw,
            args=self.args,
            timings=self.timings,
            title=reporting.CLI_RESULTS_SUMMARY_TITLE if title is None else title,
            include_sequences=include_sequences,
            include_timings=include_timings,
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


@dataclass(slots=True)
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
        print_text(
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
        print_text(
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


@dataclass(slots=True)
class GenerateResult:
    benchmark: str | None
    source: Any
    cache_dir: Path
    detectors: tuple[Path, ...]
    reid_models: tuple[Path, ...]
    timings: dict[str, Any] = field(default_factory=dict)
    args: Any = None

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"GenerateResult(benchmark={self.benchmark!r}, source={self.source!r}, "
            f"cache_dir={self.cache_dir!r}, detectors={self.detectors!r}, reid_models={self.reid_models!r})"
        )

    def render(self) -> str:
        timing_stats = reporting.timing_stats_from_snapshot(self.timings)
        if timing_stats is not None:
            summary = timing_stats.format_summary()
            if summary:
                return summary

        target = self.benchmark or self.source
        lines = ["GENERATE SUMMARY"]
        if target is not None:
            label = "Benchmark" if self.benchmark else "Source"
            lines.append(f"{label}: {target}")
        lines.append(f"Cache dir: {self.cache_dir}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print_text(self.render())

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "source": None if self.source is None else str(self.source),
            "cache_dir": str(self.cache_dir),
            "detectors": [str(path) for path in self.detectors],
            "reid_models": [str(path) for path in self.reid_models],
            "timings": dict(self.timings),
        }


@dataclass(slots=True)
class ExportResult:
    weights: Path
    files: dict[str, Any]
    parity: dict[str, dict[str, Any]] = field(default_factory=dict)
    half: bool = False

    @property
    def parity_ok(self) -> bool:
        """True iff every checked format is within tolerance.

        Formats whose parity was skipped (e.g. ``engine``) don't count
        against the result.
        """
        if not self.parity:
            return True
        return all(stats.get("ok", False) for stats in self.parity.values())

    @property
    def embedding_weights(self) -> Path:
        """Return the preferred exported model for embedding inference."""
        if "onnx" in self.files:
            return Path(self.files["onnx"])
        return Path(self.weights)

    def embed(
        self,
        *,
        source: str | Path | Any,
        boxes: Any = None,
        device: str = "cpu",
        half: bool | None = None,
        preprocess: str | None = None,
    ):
        """Generate ReID embeddings from the exported model when available."""
        import boxmot.reid as reid_module

        reid = reid_module.ReID(
            self.embedding_weights,
            device=device,
            half=self.half if half is None else bool(half),
            preprocess_name=preprocess,
        )
        return reid(source, boxes=boxes)


@dataclass(slots=True)
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

    def __iter__(self) -> Iterator[FrameResult]:
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

    def renderable(self) -> RenderableType:
        self.refresh()
        return _build_tracking_summary_renderable(self._summary)

    def print_summary(self) -> None:
        print_text(self.render())

    def refresh(self) -> None:
        summary_fn = getattr(self.results, "summary", None)
        if callable(summary_fn):
            self._summary = summary_fn()
        else:
            self._summary = _results_summary_snapshot(self.results, self.source)
        self._timings = _track_timings_from_summary(self._summary)


__all__ = (
    "ExportResult",
    "GenerateResult",
    "TrackRunResult",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
)
