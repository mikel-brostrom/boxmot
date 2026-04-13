# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

import cv2
import yaml

from boxmot.configs import BOXMOT_DEFAULTS, build_mode_namespace
from boxmot.data import IMAGE_EXTS, VIDEO_EXTS
from boxmot.engine.results import Results, Tracks
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils import configure_logging as _configure_boxmot_logging, logger as LOGGER
from boxmot.utils.compat import dataclass_slots_kwargs
from boxmot.utils.misc import increment_path, resolve_model_path
from boxmot.utils.timing import TimingStats
from boxmot.utils.torch_utils import select_device

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
CORE_SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1")
DEFAULT_VALIDATION_REPORT_TITLE = "VAL RESULTS"
DEFAULT_TUNE_BEST_REPORT_TITLE = "TUNE BEST RESULTS"
CLI_RESULTS_SUMMARY_TITLE = "📊 RESULTS SUMMARY"
CLI_TUNE_BEST_SUMMARY_TITLE = "📊 BEST TRIAL SUMMARY"
REID_TRACKERS = {"strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"}
TRACKER_CLASS_TO_NAME = {
    class_path.rsplit(".", 1)[-1].lower(): tracker_name
    for tracker_name, class_path in TRACKER_MAPPING.items()
}
SUMMARY_DISPLAY_NAMES = {
    "cls_comb_det_av": "Class Avg (Det)",
    "cls_comb_cls_av": "Class Avg (Cls)",
    "HUMAN": "Human (Super)",
    "VEHICLE": "Vehicle (Super)",
    "BIKE": "Bike (Super)",
    "all": "All Classes",
}


class _DefaultArg:
    def __repr__(self) -> str:
        return "DEFAULT"


_UNSET = _DefaultArg()


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
        return _render_validation_cli_report(
            self.raw,
            args=self.args,
            timings=self.timings,
            title=CLI_RESULTS_SUMMARY_TITLE if title is None else title,
            include_sequences=include_sequences,
            include_timings=include_timings,
        )

    def format_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        report_title = DEFAULT_VALIDATION_REPORT_TITLE if title is None else title
        return _format_validation_report(self.raw, title=report_title, include_sequences=include_sequences)

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
        payload = {
            "index": self.index,
            "config": dict(self.config),
            "score": list(self.score),
            "metrics": self.metrics.to_dict(include_raw=include_raw),
        }
        return payload


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
        return _render_validation_cli_report(
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
        print(
            self.render(
                title=title,
                include_sequences=include_sequences,
                include_timings=include_timings,
            )
        )

    def format_best_report(self, *, title: str | None = None, include_sequences: bool = True) -> str:
        report_title = DEFAULT_TUNE_BEST_REPORT_TITLE if title is None else title
        return self.best.metrics.format_report(title=report_title, include_sequences=include_sequences)

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


def _normalize_classes(classes: Any) -> list[int] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        parts = [part for part in re.split(r"[\s,]+", classes.strip()) if part]
        return [int(part) for part in parts]
    if isinstance(classes, int):
        return [int(classes)]
    return [int(value) for value in classes]


def _is_leaf_source(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    if not path.is_dir():
        return False
    img_dir = path / "img1" if (path / "img1").is_dir() else path
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS for child in img_dir.iterdir())


def _expand_sources(source: Any) -> list[Any]:
    if isinstance(source, (list, tuple)):
        return list(source)

    if not isinstance(source, (str, Path)):
        return [source]

    path = Path(source)
    if not path.is_dir() or _is_leaf_source(path):
        return [source]

    children = [child for child in sorted(path.iterdir()) if _is_leaf_source(child)]
    return children or [source]


def _coerce_results(data: Any, detector=None, reid=None, tracker=None, verbose: bool = False) -> list[Results]:
    if isinstance(data, Results):
        return [data]

    if isinstance(data, (list, tuple)) and all(isinstance(item, Results) for item in data):
        return list(data)

    if detector is None or tracker is None:
        raise ValueError("Detector and tracker are required when evaluating raw sources.")

    return [track(source, detector, reid, tracker, verbose=verbose) for source in _expand_sources(data)]


def _ensure_model_path(model_ref: str | Path | None) -> Path | None:
    if model_ref is None:
        return None
    path = Path(model_ref)
    if not path.suffix:
        path = path.with_suffix(".pt")
    return resolve_model_path(path)


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "run"


def _resolve_output_stem(source: Any) -> str:
    source_str = str(source)
    if source_str.isdigit():
        return f"camera_{source_str}"

    if "://" in source_str:
        parsed = urlparse(source_str)
        pieces = [parsed.scheme, parsed.netloc, parsed.path.strip("/")]
        return _sanitize_name("_".join(piece for piece in pieces if piece))

    path = Path(source_str)
    if path.name == "img1" and path.parent.name:
        return _sanitize_name(path.parent.name)
    if path.suffix:
        return _sanitize_name(path.stem)
    return _sanitize_name(path.name)


def _extract_summary(raw_results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results_module = import_module("boxmot.utils.evaluation.results")
    label, metrics = results_module._select_plot_metrics_data(raw_results)
    if not metrics and raw_results:
        first_value = next(iter(raw_results.values()), {})
        if isinstance(first_value, dict):
            metrics = first_value

    summary = {
        column: metrics.get(column, 0)
        for column in SUMMARY_COLUMNS
        if isinstance(metrics, dict)
    }
    return label, summary


def _timing_summary_from_stats(timing_stats: TimingStats) -> dict[str, Any]:
    totals = dict(timing_stats.totals)
    total_ms = float(totals.get("total", 0.0) or 0.0)
    if total_ms == 0.0:
        total_ms = float(sum(totals.values()))

    frames = int(timing_stats.frames)
    avg_ms = {
        key: (float(value) / frames if frames else 0.0)
        for key, value in totals.items()
    }
    avg_total_ms = total_ms / frames if frames else 0.0
    fps = (1000.0 * frames / total_ms) if total_ms else 0.0

    return {
        "frames": frames,
        "totals_ms": {**{key: float(value) for key, value in totals.items()}, "total": total_ms},
        "avg_ms": {**avg_ms, "total": avg_total_ms},
        "fps": fps,
    }


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


def _is_live_track_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isdigit() or "://" in source
    return False


def _compare_scores(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return left > right


def _core_summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(summary.get(metric, 0.0) or 0.0)
        for metric in CORE_SUMMARY_COLUMNS
    }


def _format_core_summary(summary: dict[str, Any]) -> str:
    metrics = _core_summary_metrics(summary)
    return " ".join(f"{metric}={value:.3f}" for metric, value in metrics.items())


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _display_summary_name(name: str) -> str:
    return SUMMARY_DISPLAY_NAMES.get(name, name)


def _combined_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        column: metrics.get(column, 0)
        for column in SUMMARY_COLUMNS
        if _is_numeric_metric(metrics.get(column, 0)) or column in metrics
    }


def _format_summary_cell(column: str, value: Any) -> str:
    if column in {"IDSW", "IDs"}:
        return f"{int(value or 0):>10}"
    return f"{float(value or 0):>10.2f}"


def _supports_ansi_color(stream: Any | None = None) -> bool:
    output = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return bool(hasattr(output, "isatty") and output.isatty())


def _colorize_delta(text: str, column: str, delta: float, *, colorize: bool) -> str:
    if not colorize or delta == 0:
        return text

    positive_is_better = column not in {"IDSW", "IDs"}
    is_improvement = delta > 0 if positive_is_better else delta < 0
    color_code = "32" if is_improvement else "31"
    return f"\033[{color_code}m{text}\033[0m"


def _format_summary_delta_cell(
    column: str,
    value: Any,
    baseline_value: Any | None = None,
    *,
    width: int,
    colorize: bool,
) -> str:
    if baseline_value is None:
        if column in {"IDSW", "IDs"}:
            return f"{int(value or 0):>{width}}"
        return f"{float(value or 0):>{width}.2f}"

    if column in {"IDSW", "IDs"}:
        current = int(value or 0)
        baseline = int(baseline_value or 0)
        delta = current - baseline
        delta_text = f"({delta:+d})"
        plain = f"{current} {delta_text}"
        padded = f"{plain:>{width}}"
        return padded.replace(delta_text, _colorize_delta(delta_text, column, float(delta), colorize=colorize), 1)

    current = float(value or 0.0)
    baseline = float(baseline_value or 0.0)
    delta = current - baseline
    delta_text = f"({delta:+.2f})"
    plain = f"{current:.2f} {delta_text}"
    padded = f"{plain:>{width}}"
    return padded.replace(delta_text, _colorize_delta(delta_text, column, delta, colorize=colorize), 1)


def _format_summary_table(title: str, rows: list[tuple[str, dict[str, Any], bool]], *, name_header: str = "Sequence") -> str:
    if not rows:
        return ""

    name_width = max(len(name_header), *(len(name) for name, _, _ in rows))
    header = f"{name_header:<{name_width}} " + " ".join(f"{column:>10}" for column in SUMMARY_COLUMNS)
    total_width = len(header)
    lines = [
        "=" * total_width,
        f"{title:^{total_width}}",
        "=" * total_width,
        header,
        "-" * total_width,
    ]

    for row_name, metrics, _highlight in rows:
        values = " ".join(_format_summary_cell(column, metrics.get(column, 0)) for column in SUMMARY_COLUMNS)
        lines.append(f"{row_name:<{name_width}} {values}")

    lines.append("=" * total_width)
    return "\n".join(lines)


def _iter_validation_sections(raw: dict[str, Any], *, include_sequences: bool = True) -> list[tuple[str, list[tuple[str, dict[str, Any], bool]]]]:
    if not raw:
        return []

    flat_summary = _combined_summary_metrics(raw)
    if flat_summary:
        rows: list[tuple[str, dict[str, Any], bool]] = []
        if include_sequences:
            per_sequence = raw.get("per_sequence", {})
            if isinstance(per_sequence, dict):
                rows.extend(
                    (seq_name, metrics, False)
                    for seq_name, metrics in sorted(per_sequence.items())
                    if isinstance(metrics, dict)
                )
        rows.append(("COMBINED", flat_summary, True))
        return [("Combined", rows)]

    sections: list[tuple[str, list[tuple[str, dict[str, Any], bool]]]] = []
    for section_name, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue

        rows = []
        if include_sequences:
            per_sequence = metrics.get("per_sequence", {})
            if isinstance(per_sequence, dict):
                rows.extend(
                    (seq_name, seq_metrics, False)
                    for seq_name, seq_metrics in sorted(per_sequence.items())
                    if isinstance(seq_metrics, dict)
                )

        combined = _combined_summary_metrics(metrics)
        if not rows and not combined:
            continue
        rows.append(("COMBINED", combined or metrics, True))
        sections.append((_display_summary_name(section_name), rows))

    return sections


def _format_validation_report(raw: dict[str, Any], *, title: str | None = None, include_sequences: bool = True) -> str:
    sections = _iter_validation_sections(raw, include_sequences=include_sequences)
    if not sections:
        fallback_title = title or "Results"
        return f"{fallback_title}\n{_format_core_summary(raw if isinstance(raw, dict) else {})}"

    if len(sections) == 1:
        section_title, rows = sections[0]
        return _format_summary_table(title or section_title, rows)

    blocks = []
    for section_title, rows in sections:
        block_title = section_title if title is None else f"{title} | {section_title}"
        blocks.append(_format_summary_table(block_title, rows))
    return "\n\n".join(block for block in blocks if block)


def _render_cli_summary_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, dict[str, Any], bool, dict[str, Any] | None]],
    *,
    total_width: int,
    name_width: int,
    colorize: bool,
) -> str:
    if not rows:
        return ""

    compare_enabled = any(compare_metrics is not None for _, _, _, compare_metrics in rows)
    cell_width = 16 if compare_enabled else 10
    header_values = [name_header, *SUMMARY_COLUMNS]
    header_fmt = f"{{:<{name_width}}} " + " ".join([f"{{:>{cell_width}}}"] * len(SUMMARY_COLUMNS))
    lines = [
        "=" * total_width,
        f"{title:^{total_width}}",
        "=" * total_width,
        header_fmt.format(*header_values),
        "-" * total_width,
    ]
    for row_name, metrics, _highlight, compare_metrics in rows:
        vals_str = " ".join(
            _format_summary_delta_cell(
                column,
                metrics.get(column, 0),
                None if compare_metrics is None else compare_metrics.get(column),
                width=cell_width,
                colorize=colorize,
            )
            for column in SUMMARY_COLUMNS
        )
        lines.append(f"{row_name:<{name_width}} {vals_str}")
    lines.append("=" * total_width)
    return "\n".join(lines)


def _load_report_cfg(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    try:
        benchmark_module = import_module("boxmot.data.benchmark")
        return benchmark_module.load_benchmark_cfg_from_args(args) or {}
    except Exception:
        return {}


def _infer_single_class_report_name(args: Any) -> str:
    if args is not None:
        remapped = getattr(args, "remapped_class_names", None)
        if remapped:
            return str(remapped[0])

        translated = getattr(args, "translated_benchmark_class_names", None)
        if translated:
            return str(translated[0])

        cfg = _load_report_cfg(args)
        bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
        eval_classes = bench_cfg.get("eval_classes")
        if isinstance(eval_classes, dict) and len(eval_classes) == 1:
            return str(next(iter(eval_classes.values())))
        if isinstance(eval_classes, (list, tuple)) and len(eval_classes) == 1:
            return str(eval_classes[0])

        class_indices = getattr(args, "classes", None)
        if class_indices is not None:
            indices = class_indices if isinstance(class_indices, list) else [class_indices]
            if len(indices) == 1:
                benchmark_module = import_module("boxmot.data.benchmark")
                return str(benchmark_module.COCO_CLASSES[int(indices[0])])

    return "results"


def _normalize_report_results(raw: dict[str, Any], args: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        return {}

    if _combined_summary_metrics(raw):
        return {_infer_single_class_report_name(args): raw}

    normalized: dict[str, dict[str, Any]] = {}
    for name, metrics in raw.items():
        if isinstance(metrics, dict):
            normalized[str(name)] = metrics
    return normalized


def _timing_stats_from_snapshot(timings: dict[str, Any] | None) -> TimingStats | None:
    if not timings:
        return None

    totals_ms = timings.get("totals_ms")
    if not isinstance(totals_ms, dict):
        return None

    timing_stats = TimingStats()
    for key in timing_stats.totals:
        timing_stats.totals[key] = float(totals_ms.get(key, 0.0) or 0.0)
    timing_stats.frames = int(timings.get("frames", 0) or 0)
    return timing_stats


def _render_validation_cli_report(
    raw: dict[str, Any],
    *,
    args: Any = None,
    timings: dict[str, Any] | None = None,
    title: str = CLI_RESULTS_SUMMARY_TITLE,
    include_sequences: bool = True,
    include_timings: bool = False,
    compare_raw: dict[str, Any] | None = None,
    compare_args: Any = None,
    colorize: bool | None = None,
) -> str:
    if colorize is None:
        colorize = _supports_ansi_color()

    results_module = import_module("boxmot.utils.evaluation.results")
    parsed_results = _normalize_report_results(raw, args)
    if not parsed_results:
        return _format_core_summary(raw if isinstance(raw, dict) else {})

    compare_results = _normalize_report_results(compare_raw, compare_args) if compare_raw else {}

    cfg = _load_report_cfg(args)
    primary_keys, aggregate_keys = results_module._summary_sort_keys(parsed_results, args or object(), cfg)
    if not primary_keys and not aggregate_keys:
        primary_keys = list(parsed_results.keys())

    single_sequence = all(
        len(metrics.get("per_sequence", {})) <= 1 for metrics in parsed_results.values() if isinstance(metrics, dict)
    )

    all_names = [results_module._display_summary_name(name) for name in [*primary_keys, *aggregate_keys]]
    for class_metrics in parsed_results.values():
        all_names.extend(class_metrics.get("per_sequence", {}).keys())
    all_names.extend([f"COMBINED ({results_module._display_summary_name(name)})" for name in primary_keys])

    name_width = max(18, max((len(name) for name in all_names), default=18) + 2)
    cell_width = 16 if compare_results else 10
    total_width = name_width + 1 + (cell_width * len(SUMMARY_COLUMNS)) + (len(SUMMARY_COLUMNS) - 1)

    blocks = [
        "\n".join(
            [
                "=" * total_width,
                f"{title:^{total_width}}",
                "=" * total_width,
            ]
        )
    ]

    if len(primary_keys) > 1:
        class_rows = [
            (
                results_module._display_summary_name(name),
                parsed_results[name],
                False,
                compare_results.get(name),
            )
            for name in primary_keys
        ]
        blocks.append(
            _render_cli_summary_table(
                "Per-Class Combined Metrics",
                "Class",
                class_rows,
                total_width=total_width,
                name_width=name_width,
                colorize=bool(colorize),
            )
        )

        if aggregate_keys:
            aggregate_rows = [
                (
                    results_module._display_summary_name(name),
                    parsed_results[name],
                    False,
                    compare_results.get(name),
                )
                for name in aggregate_keys
            ]
            blocks.append(
                _render_cli_summary_table(
                    "Aggregate Groups",
                    "Group",
                    aggregate_rows,
                    total_width=total_width,
                    name_width=name_width,
                    colorize=bool(colorize),
                )
            )

        if include_sequences and not single_sequence:
            for class_name in primary_keys:
                compare_class_metrics = compare_results.get(class_name, {})
                per_sequence_rows = [
                    (
                        seq_name,
                        seq_metrics,
                        False,
                        compare_class_metrics.get("per_sequence", {}).get(seq_name)
                        if isinstance(compare_class_metrics, dict)
                        else None,
                    )
                    for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
                ]
                per_sequence_rows.append(
                    (
                        f"COMBINED ({results_module._display_summary_name(class_name)})",
                        parsed_results[class_name],
                        True,
                        compare_class_metrics if isinstance(compare_class_metrics, dict) else None,
                    )
                )
                blocks.append(
                    _render_cli_summary_table(
                        f"Per-Sequence Details: {results_module._display_summary_name(class_name)}",
                        "Sequence",
                        per_sequence_rows,
                        total_width=total_width,
                        name_width=name_width,
                        colorize=bool(colorize),
                    )
                )
    else:
        detail_keys = primary_keys or aggregate_keys or list(parsed_results.keys())
        for class_name in detail_keys:
            compare_class_metrics = compare_results.get(class_name, {})
            per_sequence_rows = [
                (
                    seq_name,
                    seq_metrics,
                    False,
                    compare_class_metrics.get("per_sequence", {}).get(seq_name)
                    if isinstance(compare_class_metrics, dict)
                    else None,
                )
                for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
            ]
            if not include_sequences:
                per_sequence_rows = []
            if not single_sequence or not per_sequence_rows:
                per_sequence_rows.append(
                    (
                        f"COMBINED ({results_module._display_summary_name(class_name)})",
                        parsed_results[class_name],
                        True,
                        compare_class_metrics if isinstance(compare_class_metrics, dict) else None,
                    )
                )
            blocks.append(
                _render_cli_summary_table(
                    results_module._display_summary_name(class_name),
                    "Sequence",
                    per_sequence_rows,
                    total_width=total_width,
                    name_width=name_width,
                    colorize=bool(colorize),
                )
            )

    if include_timings:
        timing_stats = _timing_stats_from_snapshot(timings)
        if timing_stats is not None:
            timing_summary = timing_stats.format_summary()
            if timing_summary:
                blocks.append(timing_summary)

    return "\n".join(block for block in blocks if block)


def _print_validation_cli_report(
    raw: dict[str, Any],
    *,
    args: Any = None,
    timings: dict[str, Any] | None = None,
    title: str = CLI_RESULTS_SUMMARY_TITLE,
    include_sequences: bool = True,
    include_timings: bool = False,
) -> None:
    report = _render_validation_cli_report(
        raw,
        args=args,
        timings=timings,
        title=title,
        include_sequences=include_sequences,
        include_timings=include_timings,
    )
    if report:
        print(report)


def _format_remaining_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"

    total_seconds = 0 if seconds <= 0 else int(math.ceil(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _estimate_tune_remaining(trial_durations: Sequence[float], remaining_trials: int) -> float | None:
    if remaining_trials <= 0:
        return 0.0
    if not trial_durations:
        return None
    avg_trial_seconds = sum(trial_durations) / len(trial_durations)
    return avg_trial_seconds * remaining_trials


def _format_progress_bar(current: int, total: int, *, bar_width: int = 20) -> tuple[str, float]:
    if total <= 0:
        pct = 1.0 if current >= total else 0.0
    else:
        pct = min(max(current / total, 0.0), 1.0)

    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    return bar, pct


def _format_named_progress(label: str, current: int, total: int, *, detail: str = "") -> str:
    bar, pct = _format_progress_bar(current, total)
    message = f"  {label:<8s} {bar} {pct:>5.0%}  ({current}/{total})"
    if detail:
        message = f"{message}  {detail}"
    return message


def _format_tune_progress(
    completed: int,
    total: int,
    summary: dict[str, Any] | None = None,
    *,
    current_trial: int | None = None,
    is_new_best: bool = False,
    remaining_seconds: float | None = None,
) -> str:
    remaining = _format_remaining_time(remaining_seconds)
    if summary is None:
        running = current_trial if current_trial is not None else (completed + 1)
        return _format_named_progress(
            "Tune",
            completed,
            total,
            detail=f"running trial {running}/{total}  remaining {remaining}",
        )

    summary_prefix = "last " if current_trial is not None and completed < total else ""
    summary_text = f"{summary_prefix}{_format_core_summary(summary)}"
    status = "  best" if is_new_best else ""
    if current_trial is not None and completed < total:
        return _format_named_progress(
            "Tune",
            completed,
            total,
            detail=(
                f"running trial {current_trial}/{total}  "
                f"{summary_text}{status}  remaining {remaining}"
            ),
        )

    return _format_named_progress(
        "Tune",
        completed,
        total,
        detail=f"{summary_text}{status}  remaining {remaining}",
    )


def _write_progress_line(
    message: str,
    previous_width: int,
    *,
    stream=None,
    final: bool = False,
) -> int:
    output = sys.stderr if stream is None else stream
    width = max(previous_width, len(message))
    is_tty = hasattr(output, "isatty") and output.isatty()
    rendered = f"\033[36m{message}\033[0m" if is_tty else message
    prefix = "\r\033[2K" if is_tty else "\r"
    output.write(prefix + rendered + (" " * (width - len(message))))
    if final:
        output.write("\n")
    output.flush()
    return width


def _flush_logger_queue() -> None:
    complete = getattr(LOGGER, "complete", None)
    if complete is None:
        return
    complete()


@contextmanager
def _suppress_boxmot_logs(enabled: bool, *, level: str = "WARNING"):
    if not enabled:
        yield
        return

    LOGGER.remove()
    LOGGER.add(
        sys.stderr,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    try:
        yield
    finally:
        _configure_boxmot_logging(main_only=True)


class _TrackerReIDAdapter:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, inputs, boxes=None, **_kwargs):
        if boxes is None:
            raise TypeError("boxes are required when reusing a tracker ReID backend")
        return self.backend.get_features(boxes, inputs)


def track(source, detector, reid=None, tracker=None, *, verbose: bool = True, drawer=None) -> Results:
    """Create a lazy streaming tracking result iterator."""
    if tracker is None:
        raise ValueError("A tracker instance is required.")
    return Results(source, detector, reid, tracker, verbose=verbose, drawer=drawer)


def evaluate(data, detector=None, reid=None, tracker=None, *, metrics: bool = True, speed: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Aggregate run metrics over one or more tracking results.

    This helper summarizes execution counts and timing. It does not replace
    TrackEval ground-truth benchmark evaluation.
    """
    runs = _coerce_results(data, detector=detector, reid=reid, tracker=tracker, verbose=verbose)
    summaries = [run.summary() for run in runs]

    total_frames = sum(summary["frames"] for summary in summaries)
    total_detections = sum(summary["detections"] for summary in summaries)
    total_tracks = sum(summary["tracks"] for summary in summaries)
    total_det_ms = sum(summary["timings_ms"]["det"] for summary in summaries)
    total_reid_ms = sum(summary["timings_ms"]["reid"] for summary in summaries)
    total_track_ms = sum(summary["timings_ms"]["track"] for summary in summaries)
    total_ms = sum(summary["timings_ms"]["total"] for summary in summaries)

    response: dict[str, Any] = {
        "sources": len(summaries),
        "runs": summaries,
    }

    if metrics:
        response["metrics"] = {
            "frames": total_frames,
            "detections": total_detections,
            "tracks": total_tracks,
            "avg_tracks_per_frame": (total_tracks / total_frames) if total_frames else 0.0,
        }

    if speed:
        response["speed"] = {
            "det_ms": total_det_ms,
            "reid_ms": total_reid_ms,
            "track_ms": total_track_ms,
            "total_ms": total_ms,
            "avg_total_ms": (total_ms / total_frames) if total_frames else 0.0,
            "fps": (1000.0 * total_frames / total_ms) if total_ms else 0.0,
        }

    return response


class Boxmot:
    def __init__(
        self,
        detector: Any = _UNSET,
        reid: Any = _UNSET,
        tracker: Any = _UNSET,
        classes: Any = None,
        project: str | Path = BOXMOT_DEFAULTS.track.project,
    ) -> None:
        self._detector_explicit = detector is not _UNSET and detector is not None
        self._reid_explicit = reid is not _UNSET and reid is not None
        self._tracker_explicit = tracker is not _UNSET and tracker is not None

        self.detector = BOXMOT_DEFAULTS.shared.detector if detector is _UNSET else detector
        self.reid = BOXMOT_DEFAULTS.shared.reid if reid is _UNSET else reid
        self.tracker = BOXMOT_DEFAULTS.track.tracker if tracker is _UNSET else tracker
        self.classes = _normalize_classes(classes)
        self.project = Path(project)

    def _detector_path(self, required: bool = True) -> Path | None:
        spec = self.detector
        if spec is None:
            if required:
                raise ValueError("A detector model path is required for this operation.")
            return None
        if isinstance(spec, (str, Path)):
            return _ensure_model_path(spec)
        path = getattr(spec, "path", None)
        if path is not None:
            return _ensure_model_path(path)
        if required:
            raise ValueError("Detector benchmark workflows require a detector with a resolvable .path.")
        return None

    def _reid_path(self, required: bool = True) -> Path | None:
        spec = self.reid
        if spec is None:
            if required:
                raise ValueError("A ReID model path is required for this operation.")
            return None
        if isinstance(spec, (str, Path)):
            return _ensure_model_path(spec)
        path = getattr(spec, "path", None) or getattr(spec, "weights", None)
        if path is not None:
            return _ensure_model_path(path)
        if required:
            raise ValueError("This operation requires a ReID model with a resolvable .path or .weights.")
        return None

    def _tracker_name(self, required: bool = True) -> str | None:
        spec = self.tracker
        if spec is None:
            if required:
                raise ValueError("A tracker is required.")
            return None
        if isinstance(spec, str):
            name = spec.lower()
            if name in TRACKER_MAPPING:
                return name
        class_name = spec.__class__.__name__.lower() if spec is not None else ""
        if class_name in TRACKER_CLASS_TO_NAME:
            return TRACKER_CLASS_TO_NAME[class_name]
        if required:
            raise ValueError("Could not infer a registered tracker name from the provided tracker spec.")
        return None

    def _tracker_config_from_spec(self) -> dict[str, Any] | None:
        if isinstance(self.tracker, str) or self.tracker is None:
            return None

        tracker_name = self._tracker_name(required=False)
        if tracker_name is None:
            return None

        with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        resolved: dict[str, Any] = {}
        for key, details in config.items():
            if hasattr(self.tracker, key):
                resolved[key] = getattr(self.tracker, key)
            else:
                resolved[key] = details.get("default")
        return resolved

    def _build_detector(self, *, device: str = BOXMOT_DEFAULTS.track.device, imgsz=None, conf=None, iou=BOXMOT_DEFAULTS.track.iou):
        from boxmot.detectors import Detector as PublicDetector

        spec = self.detector
        if isinstance(spec, (str, Path)):
            return PublicDetector(
                path=_ensure_model_path(spec),
                device=device,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=self.classes,
            )

        current_device = getattr(spec, "device", None)
        if current_device is not None and str(current_device) != str(device):
            raise ValueError(
                f"Detector instance is already bound to device '{current_device}'. "
                f"Create it on '{device}' or pass a path/string detector spec instead."
            )

        if imgsz is not None and hasattr(spec, "imgsz"):
            spec.imgsz = imgsz
        if conf is not None and hasattr(spec, "conf"):
            spec.conf = float(conf)
        if iou is not None and hasattr(spec, "iou"):
            spec.iou = float(iou)
        if self.classes is not None and hasattr(spec, "classes"):
            spec.classes = self.classes
        return spec

    def _build_reid(self, *, device: str = BOXMOT_DEFAULTS.track.device, half: bool = BOXMOT_DEFAULTS.track.half):
        from boxmot.reid import ReID as PublicReID

        if self.reid is None:
            return None

        spec = self.reid
        if isinstance(spec, (str, Path)):
            return PublicReID(_ensure_model_path(spec), device=device, half=half)

        current_device = getattr(spec, "device", None)
        if current_device is not None and str(current_device) != str(device):
            raise ValueError(
                f"ReID instance is already bound to device '{current_device}'. "
                f"Create it on '{device}' or pass a path/string ReID spec instead."
            )
        return spec

    def _build_tracker(self, *, device: str = BOXMOT_DEFAULTS.track.device, half: bool = BOXMOT_DEFAULTS.track.half):
        if not isinstance(self.tracker, str):
            return self.tracker

        tracker_name = self._tracker_name(required=True)
        reid_path = self._reid_path(required=False)
        return create_tracker(
            tracker_type=tracker_name,
            tracker_config=get_tracker_config(tracker_name),
            reid_weights=reid_path,
            device=select_device(device),
            half=half,
            per_class=False,
        )

    def _build_track_reid(
        self,
        tracker: Any,
        *,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
    ):
        if isinstance(self.tracker, str):
            tracker_name = self._tracker_name(required=False)
            if tracker_name in REID_TRACKERS:
                if hasattr(tracker, "with_reid") and not bool(getattr(tracker, "with_reid")):
                    return None

                tracker_backend = getattr(tracker, "reid_model", None) or getattr(tracker, "model", None)
                if tracker_backend is not None:
                    return _TrackerReIDAdapter(tracker_backend)

        return self._build_reid(device=device, half=half)

    def _base_eval_args(
        self,
        benchmark: str | Path,
        *,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        show_progress: bool = True,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    ):
        reid_path = self._reid_path(required=False) or BOXMOT_DEFAULTS.shared.reid
        tracker_spec = self.tracker
        per_class = bool(getattr(tracker_spec, "per_class", False)) if not isinstance(tracker_spec, str) else False

        args = build_mode_namespace(
            "eval",
            {
                "data": str(benchmark),
                "benchmark": str(benchmark),
                "source": None,
                "split": "",
                "detector": [self._detector_path(required=True)],
                "reid": [reid_path],
                "device": device,
                "half": bool(half),
                "imgsz": imgsz,
                "conf": conf,
                "iou": float(iou),
                "classes": self.classes,
                "project": Path(project or self.project),
                "name": "python_api",
                "exist_ok": True,
                "ci": True,
                "tracker": self._tracker_name(required=True),
                "verbose": bool(verbose),
                "show_progress": bool(show_progress),
                "postprocessing": postprocessing,
                "fps": None,
                "show": False,
                "show_trajectories": False,
                "show_kf_preds": False,
                "save": False,
                "save_txt": False,
                "save_crop": False,
                "per_class": per_class,
                "target_id": None,
                "vid_stride": BOXMOT_DEFAULTS.eval.vid_stride,
                "tracking_backend": "thread",
            },
            explicit_keys={
                *({"detector"} if self._detector_explicit else set()),
                *({"reid"} if self._reid_explicit else set()),
                *({"tracker"} if self._tracker_explicit else set()),
                *({"device"} if device != BOXMOT_DEFAULTS.eval.device else set()),
                *({"half"} if bool(half) != bool(BOXMOT_DEFAULTS.eval.half) else set()),
            },
        )
        args.reid_device = device
        args.reid_half = bool(half)
        args.dataset_detector_cfg = None
        args.eval_box_type = None
        args.gt_class_remap = None
        args.gt_class_distractor_ids = None
        args.remapped_class_ids = None
        args.remapped_class_names = None
        args.translated_benchmark_class_names = None
        return args

    def _run_validation_pipeline(
        self,
        *,
        benchmark: str | Path,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        show_progress: bool = True,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
        evolve_config: dict[str, Any] | None = None,
    ) -> ValidationResult:
        evaluator = import_module("boxmot.engine.evaluator")
        replay = import_module("boxmot.engine.replay")
        args = self._base_eval_args(
            benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=show_progress,
            postprocessing=postprocessing,
        )

        timing_stats = TimingStats()
        evaluator.eval_setup(args)
        evaluator.run_generate_dets_embs(args, timing_stats=timing_stats)
        tracker_config = evolve_config if evolve_config is not None else self._tracker_config_from_spec()
        replay.run_generate_mot_results(
            args,
            evolve_config=tracker_config,
            timing_stats=timing_stats,
            quiet=not show_progress,
        )
        raw_results = evaluator.run_trackeval(args, verbose=verbose)
        summary_label, summary = _extract_summary(raw_results)

        return ValidationResult(
            benchmark=str(benchmark),
            raw=raw_results,
            summary_label=summary_label,
            summary=summary,
            exp_dir=getattr(args, "exp_dir", None),
            timings=_timing_summary_from_stats(timing_stats),
            args=args,
        )

    def _load_tracker_search_space(self) -> dict[str, Any]:
        tracker_name = self._tracker_name(required=True)
        with open(get_tracker_config(tracker_name), "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _default_tracker_config(self) -> dict[str, Any]:
        existing = self._tracker_config_from_spec()
        if existing is not None:
            return existing

        search_space = self._load_tracker_search_space()
        return {
            key: details.get("default")
            for key, details in search_space.items()
        }

    def _sample_param(self, spec: dict[str, Any], rng: random.Random):
        param_type = str(spec.get("type", "choice")).lower()

        if param_type == "uniform":
            low, high = spec["range"]
            return float(rng.uniform(float(low), float(high)))

        if param_type == "loguniform":
            low, high = spec["range"]
            return float(math.exp(rng.uniform(math.log(float(low)), math.log(float(high)))))

        if param_type == "randint":
            low, high = spec["range"]
            return int(rng.randint(int(low), int(high)))

        if param_type == "qrandint":
            low, high, step = spec["range"]
            choices = list(range(int(low), int(high), int(step)))
            return int(rng.choice(choices))

        if param_type in {"choice", "grid_search"}:
            options = spec.get("options") or spec.get("values") or []
            if not options:
                return spec.get("default")
            return rng.choice(list(options))

        return spec.get("default")

    def _iter_tune_configs(self, n_trials: int, rng: random.Random) -> Iterator[dict[str, Any]]:
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")

        search_space = self._load_tracker_search_space()
        yield self._default_tracker_config()

        for _ in range(n_trials - 1):
            yield {
                key: self._sample_param(details, rng)
                for key, details in search_space.items()
            }

    @staticmethod
    def _score_summary(
        summary: dict[str, Any],
        maximize: Sequence[str],
        minimize: Sequence[str],
    ) -> tuple[float, ...]:
        score: list[float] = []
        for metric in maximize:
            score.append(float(summary.get(metric, float("-inf"))))
        for metric in minimize:
            score.append(-float(summary.get(metric, float("inf"))))
        return tuple(score)

    def _resolve_track_output_dir(self, source: Any) -> Path:
        base = self.project / "track" / _resolve_output_stem(source)
        return increment_path(base, mkdir=True)

    def _resolve_output_fps(self, source: Any, fallback: float = 30.0) -> float:
        if isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.isdigit() or "://" in source_str:
                return fallback
            path = Path(source_str)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                capture = cv2.VideoCapture(str(path))
                try:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                finally:
                    capture.release()
                if fps and fps > 0:
                    return float(fps)
        return fallback

    def _save_video(self, results: Results, video_path: Path, fps: float) -> Path:
        frames = results.materialize()
        if not frames:
            return video_path

        height, width = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for track_result in frames:
                writer.write(track_result.render())
        finally:
            writer.release()
        return video_path

    def _run_export_pipeline(
        self,
        *,
        include: Sequence[str],
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = BOXMOT_DEFAULTS.export.dynamic,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
    ) -> ExportResult:
        export_module = import_module("boxmot.engine.export")
        weights = self._reid_path(required=True)
        args = build_mode_namespace(
            "export",
            {
                "weights": weights,
                "include": tuple(include),
                "device": device,
                "half": bool(half),
                "optimize": bool(optimize),
                "dynamic": bool(dynamic),
                "simplify": bool(simplify),
                "opset": int(opset),
                "workspace": int(workspace),
                "verbose": bool(verbose),
                "batch_size": int(batch_size),
                "imgsz": imgsz,
            },
            explicit_keys={"weights", "device", "half", "optimize", "dynamic", "simplify", "opset", "workspace", "batch_size", "imgsz", "include"},
        )
        model, dummy_input = export_module.setup_model(args)
        export_tasks = export_module.create_export_tasks(args, model, dummy_input)
        files = export_module.perform_exports(export_tasks)
        return ExportResult(weights=args.weights, files=files)

    def track(
        self,
        *,
        source: Any,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.track.iou,
        device: str = BOXMOT_DEFAULTS.track.device,
        half: bool = BOXMOT_DEFAULTS.track.half,
        save: bool = BOXMOT_DEFAULTS.track.save,
        save_txt: bool = BOXMOT_DEFAULTS.track.save_txt,
        show: bool = BOXMOT_DEFAULTS.track.show,
        drawer=None,
        verbose: bool = BOXMOT_DEFAULTS.track.verbose,
    ) -> TrackRunResult:
        with _suppress_boxmot_logs(not verbose, level="WARNING"):
            detector = self._build_detector(device=device, imgsz=imgsz, conf=conf, iou=iou)
            tracker = self._build_tracker(device=device, half=half)
            reid = self._build_track_reid(tracker, device=device, half=half)
        run = track(source, detector, reid, tracker, verbose=verbose, drawer=drawer)

        output_dir = self._resolve_track_output_dir(source)
        text_path = output_dir / "tracks.txt" if save_txt else None
        video_path = output_dir / "tracks.mp4" if save else None

        if text_path is not None:
            run.save(text_path)
        if video_path is not None:
            self._save_video(run, video_path, fps=self._resolve_output_fps(source))

        result = TrackRunResult(
            source=source,
            results=run,
            video_path=video_path,
            text_path=text_path,
        )
        if show:
            result.show()
        return result

    def val(
        self,
        *,
        benchmark: str | Path,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        postprocessing: str = BOXMOT_DEFAULTS.eval.postprocessing,
    ) -> ValidationResult:
        return self._run_validation_pipeline(
            benchmark=benchmark,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            project=project,
            verbose=verbose,
            show_progress=True,
            postprocessing=postprocessing,
        )

    def tune(
        self,
        *,
        benchmark: str | Path,
        n_trials: int = BOXMOT_DEFAULTS.tune.n_trials,
        imgsz=None,
        conf=None,
        iou: float = BOXMOT_DEFAULTS.eval.iou,
        device: str = BOXMOT_DEFAULTS.eval.device,
        half: bool = BOXMOT_DEFAULTS.eval.half,
        project: str | Path | None = None,
        maximize: Sequence[str] = BOXMOT_DEFAULTS.tune.maximize,
        minimize: Sequence[str] = BOXMOT_DEFAULTS.tune.minimize,
        verbose: bool = BOXMOT_DEFAULTS.eval.verbose,
        seed: int = 0,
    ) -> TuneResult:
        rng = random.Random(seed)
        tracker_name = self._tracker_name(required=True)
        trials: list[TuneTrialResult] = []
        best: TuneTrialResult | None = None
        progress_width = 0
        last_summary: dict[str, Any] | None = None
        last_was_best = False
        trial_durations: list[float] = []

        for index, config in enumerate(self._iter_tune_configs(n_trials, rng), start=1):
            remaining_seconds = _estimate_tune_remaining(trial_durations, n_trials - (index - 1))
            progress_width = _write_progress_line(
                _format_tune_progress(
                    index - 1,
                    n_trials,
                    summary=last_summary,
                    current_trial=index,
                    is_new_best=last_was_best,
                    remaining_seconds=remaining_seconds,
                ),
                progress_width,
            )

            trial_started = time.perf_counter()
            with _suppress_boxmot_logs(not verbose, level="WARNING"):
                metrics = self._run_validation_pipeline(
                    benchmark=benchmark,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    half=half,
                    project=project,
                    verbose=False,
                    show_progress=False,
                    evolve_config=config,
                )

            trial_durations.append(time.perf_counter() - trial_started)
            score = self._score_summary(metrics.summary, maximize=maximize, minimize=minimize)
            trial_result = TuneTrialResult(index=index, config=config, metrics=metrics, score=score)
            trials.append(trial_result)

            is_new_best = best is None or _compare_scores(trial_result.score, best.score)
            if is_new_best:
                best = trial_result
            last_summary = metrics.summary
            last_was_best = is_new_best

            remaining_seconds = _estimate_tune_remaining(trial_durations, n_trials - index)
            progress_width = _write_progress_line(
                _format_tune_progress(
                    index,
                    n_trials,
                    metrics.summary,
                    is_new_best=is_new_best,
                    remaining_seconds=remaining_seconds,
                ),
                progress_width,
                final=index == n_trials,
            )

        if best is None:
            raise RuntimeError("Tune did not produce any trials.")

        output_dir = increment_path(Path(project or self.project) / "tune" / f"{benchmark}_{tracker_name}", mkdir=True)
        best_yaml = output_dir / "best.yaml"
        with open(best_yaml, "w", encoding="utf-8") as handle:
            yaml.safe_dump(best.config, handle, sort_keys=False)

        return TuneResult(
            benchmark=str(benchmark),
            tracker=tracker_name,
            trials=trials,
            best=best,
            best_config=best.config,
            best_yaml=best_yaml,
        )

    def export(
        self,
        *,
        include: Sequence[str] = BOXMOT_DEFAULTS.export.include,
        device: str = BOXMOT_DEFAULTS.export.device,
        half: bool = BOXMOT_DEFAULTS.export.half,
        optimize: bool = BOXMOT_DEFAULTS.export.optimize,
        dynamic: bool = BOXMOT_DEFAULTS.export.dynamic,
        simplify: bool = BOXMOT_DEFAULTS.export.simplify,
        opset: int = BOXMOT_DEFAULTS.export.opset,
        workspace: int = BOXMOT_DEFAULTS.export.workspace,
        verbose: bool = False,
        batch_size: int = BOXMOT_DEFAULTS.export.batch_size,
        imgsz=None,
    ) -> ExportResult:
        return self._run_export_pipeline(
            include=include,
            device=device,
            half=half,
            optimize=optimize,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            workspace=workspace,
            verbose=verbose,
            batch_size=batch_size,
            imgsz=imgsz,
        )

__all__ = (
    "Boxmot",
    "ExportResult",
    "Results",
    "TrackRunResult",
    "Tracks",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
    "evaluate",
    "track",
)
