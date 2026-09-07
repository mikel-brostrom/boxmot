"""Rich workflow reporter for the ``track`` command."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.tracking.timing import (
    TRACK_STARTUP_TIMING_COMPONENTS,
    normalize_track_runtime_timings_ms,
    normalize_track_startup_timings_ms,
)
from boxmot.engine.ui.workflow.fields import compact_model_name, image_size_text, panel_field
from boxmot.engine.ui.workflow.reporting import RichWorkflowReporter
from boxmot.engine.ui.workflow.steps import (
    SETUP as TRACK_SETUP_STEP,
)
from boxmot.engine.ui.workflow.steps import (
    TRACK as TRACK_RUN_STEP,
)
from boxmot.engine.ui.workflow.steps import (
    TRACK_STEPS,
)


def _tracker_name_from_spec_safe(spec: Any) -> str | None:
    """Best-effort display name extraction without importing domain packages."""
    if isinstance(spec, str):
        return spec
    name = getattr(spec, "name", None)
    return name if isinstance(name, str) and name else None


def _build_track_workflow_fields(args: Any) -> list[tuple[str, object]]:
    """Build workflow fields as compact subsystem cards (like eval view)."""
    fields: list[tuple[str, object]] = []

    # ── Tracker card ──────────────────────────────────────────────
    tracker = getattr(args, "tracker", None)
    tracker_backend = getattr(args, "tracker_backend", None)
    cmc_method = getattr(args, "cmc_method", None)

    tracker_items: list[tuple[str, object]] = []
    if tracker:
        tracker_items.append(("Name", _tracker_name_from_spec_safe(tracker) or tracker))
    if tracker_backend not in {None, ""}:
        tracker_items.append(("Backend", tracker_backend))
    if cmc_method not in {None, "", "none"}:
        tracker_items.append(("CMC", cmc_method))
    if tracker_items:
        fields.append(panel_field("Tracker", tracker_items))

    # ── Detector card ─────────────────────────────────────────────
    detector = getattr(args, "detector", None)
    detector_items: list[tuple[str, object]] = []
    if detector:
        detector_items.append(("Model", compact_model_name(detector)))
    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        detector_items.append(("Size", image_size_text(imgsz)))
    conf = getattr(args, "conf", None)
    if conf is not None:
        detector_items.append(("Conf", f"≥ {conf}"))
    if detector_items:
        fields.append(panel_field("Detector", detector_items))

    # ── ReID card ─────────────────────────────────────────────────
    reid = getattr(args, "reid", None)
    reid_items: list[tuple[str, object]] = []
    if reid:
        reid_items.append(("Model", compact_model_name(reid)))
    if reid_items:
        fields.append(panel_field("ReID", reid_items))

    # ── Source card ───────────────────────────────────────────────
    source = getattr(args, "source", None)
    source_items: list[tuple[str, object]] = []
    if source not in {None, ""}:
        source_items.append(("Input", source))
    if source_items:
        fields.append(panel_field("Source", source_items))

    # ── Runtime card ──────────────────────────────────────────────
    runtime_items: list[tuple[str, object]] = []
    device = getattr(args, "device", None)
    if device not in {None, ""}:
        runtime_items.append(("Device", device))
    runtime_items.append(("Precision", "fp16" if bool(getattr(args, "half", False)) else "fp32"))
    iou = getattr(args, "iou", None)
    if iou is not None:
        runtime_items.append(("IoU", iou))
    runtime_items.append(("Show", bool(getattr(args, "show", False))))
    runtime_items.append(("Save video", bool(getattr(args, "save", False))))
    runtime_items.append(("Save txt", bool(getattr(args, "save_txt", False))))
    if runtime_items:
        fields.append(panel_field("Runtime", runtime_items))

    return fields


def _summary_mapping(summary: Any, *names: str) -> Mapping[str, Any]:
    """Read one mapping from a run summary without coupling to its class."""

    for name in names:
        value = getattr(summary, name, None)
        if isinstance(value, Mapping):
            return value
    return {}


def _summary_counter(summary: Any, name: str, *aliases: str) -> int:
    counters = _summary_mapping(summary, "counters")
    for key in (name, *aliases):
        value = getattr(summary, key, None)
        if value is None:
            value = counters.get(key)
        if value is None:
            continue
        if isinstance(value, (set, frozenset, tuple, list)):
            return len(value)
        return max(int(value), 0)
    return 0


def _timed_frame_count(summary: Any, completed_frames: int) -> int:
    """Return the timing denominator, including one partial frame if present."""

    timings = getattr(summary, "timings", None)
    if isinstance(timings, (tuple, list)) and timings:
        return len(timings)
    return completed_frames


def _format_ms(value: float, *, decimals: int, unit: bool = False) -> str:
    """Format milliseconds without rounding a positive measurement to zero."""

    value = max(float(value or 0.0), 0.0)
    suffix = " ms" if unit else ""
    if value == 0.0:
        return "—"
    threshold = 10.0**-decimals
    if value < threshold:
        return f"<{threshold:.{decimals}f}{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _format_duration_ms(value: float) -> str:
    """Format a headline duration using compact, human-scale units."""

    value = max(float(value or 0.0), 0.0)
    if value == 0.0:
        return "—"
    if value < 1_000.0:
        return _format_ms(value, decimals=1, unit=True)
    seconds = value / 1_000.0
    return f"{seconds:,.3f} s" if seconds < 10.0 else f"{seconds:,.2f} s"


def _format_fps(avg_ms: float) -> str:
    """Format stage-equivalent throughput; zero latency has no defined FPS."""

    avg_ms = max(float(avg_ms or 0.0), 0.0)
    if avg_ms == 0.0:
        return "—"
    fps = 1_000.0 / avg_ms
    if fps >= 1_000_000.0:
        return f"{fps / 1_000_000.0:,.1f}M"
    if fps >= 1_000.0:
        return f"{fps / 1_000.0:,.1f}k"
    if fps < 0.1:
        return "<0.1"
    return f"{fps:,.1f}"


def _average_ms(total_ms: float, timed_frames: int) -> float:
    return float(total_ms) / timed_frames if timed_frames else 0.0


def _count_token(value: int, singular: str, plural: str | None = None) -> Text:
    label = singular if value == 1 else (plural or f"{singular}s")
    token = Text()
    token.append(f"{value:,}", style=ui.STYLE_TEXT_STRONG)
    token.append(f" {label}", style=ui.STYLE_MUTED)
    return token


def _frame_token(completed: int, attempted: int) -> Text:
    if attempted != completed:
        token = Text()
        token.append(f"{completed:,}/{attempted:,}", style=ui.STYLE_TEXT_STRONG)
        token.append(" frames completed", style=ui.STYLE_MUTED)
        return token
    return _count_token(completed, "frame")


def _append_separator(text: Text) -> None:
    text.append("  •  ", style=ui.STYLE_SUBTLE)


def _build_status_summary(
    summary: Any,
    runtime: Mapping[str, float],
    *,
    timed_frames: int,
    compact: bool,
) -> RenderableType:
    completed = _summary_counter(summary, "frames")
    stopped = bool(getattr(summary, "interrupted", False)) or bool(getattr(summary, "stopped_by_user", False))

    status = Text()
    status.append("■ " if stopped else "✓ ", style=ui.STYLE_STATUS_ACTIVE if stopped else ui.STYLE_STATUS_DONE)
    status.append("Tracking stopped by user" if stopped else "Tracking complete", style=ui.STYLE_TEXT_STRONG)
    _append_separator(status)
    status.append(_format_duration_ms(runtime["overall"]), style=ui.STYLE_TEXT_STRONG)
    status.append(" runtime", style=ui.STYLE_MUTED)
    _append_separator(status)
    # The headline is observed run throughput, so only completed frames count.
    # Detailed stage rows deliberately use ``timed_frames`` to retain work from
    # an interrupted partial frame.
    overall_avg = _average_ms(runtime["overall"], completed)
    status.append(_format_fps(overall_avg), style=ui.STYLE_TEXT_STRONG)
    status.append(" FPS", style=ui.STYLE_MUTED)

    counters = [
        _frame_token(completed, timed_frames),
        _count_token(_summary_counter(summary, "detections"), "detection"),
        _count_token(_summary_counter(summary, "track_rows", "tracks"), "track row"),
        _count_token(_summary_counter(summary, "unique_track_ids", "unique_tracks"), "unique ID"),
        _count_token(_summary_counter(summary, "sequences"), "sequence"),
    ]
    column_count = 3 if compact else len(counters)
    grid = Table.grid(expand=True, padding=(0, 2))
    for _ in range(column_count):
        grid.add_column(ratio=1)
    for index in range(0, len(counters), column_count):
        row: list[RenderableType] = list(counters[index : index + column_count])
        row.extend(Text() for _ in range(column_count - len(row)))
        grid.add_row(*row)
    return Group(status, grid)


_STARTUP_LABELS = {
    "detector_load": "Detector load",
    "tracker_load": "Tracker load",
    "reid_encoder_load": "ReID encoder load",
    "segmentor_load": "Segmentor load",
    "pipeline_prepare": "Pipeline preparation",
    "output_prepare": "Output preparation",
    "source_open": "Source open",
    "source_first_frame": "First source frame*",
}


def _build_startup_section(timings: Mapping[str, float], *, compact: bool) -> Group:
    title = Text("Startup", style=ui.STYLE_TITLE)
    title.append(f"  {_format_duration_ms(timings['total'])} total", style=ui.STYLE_MUTED)

    pair_count = 2 if compact else 4
    table = Table.grid(expand=True, padding=(0, 1))
    for _ in range(pair_count):
        table.add_column(style=ui.STYLE_ACCENT, no_wrap=True)
        table.add_column(style=ui.STYLE_TEXT, justify="right", no_wrap=True, ratio=1)

    items = [(key, _STARTUP_LABELS[key]) for key in TRACK_STARTUP_TIMING_COMPONENTS]
    for index in range(0, len(items), pair_count):
        row: list[RenderableType | str] = []
        for key, label in items[index : index + pair_count]:
            row.extend((label, _format_duration_ms(timings[key])))
        row.extend("" for _ in range(2 * pair_count - len(row)))
        table.add_row(*row)

    note = Text("* First source frame is also counted in Source acquisition runtime.", style=ui.STYLE_MUTED)
    return Group(Rule(title, style=ui.STYLE_RULE), table, note)


def _runtime_component_rows(
    runtime: Mapping[str, float],
    *,
    include_segmentor: bool,
    include_reid: bool,
) -> list[tuple[str, tuple[tuple[str, float], ...], float]]:
    rows = [
        (
            "Detector",
            (
                ("preprocess", runtime["detector_preprocess"]),
                ("inference", runtime["detector_inference"]),
                ("postprocess", runtime["detector_postprocess"]),
            ),
            runtime["detector_total"],
        )
    ]
    if include_segmentor:
        rows.append(
            (
                "Segmentor",
                (
                    ("preprocess", runtime["segmentor_preprocess"]),
                    ("inference", runtime["segmentor_inference"]),
                    ("postprocess", runtime["segmentor_postprocess"]),
                ),
                runtime["segmentor_total"],
            )
        )
    if include_reid:
        rows.append(
            (
                "ReID encoder",
                (
                    ("preprocess", runtime["reid_preprocess"]),
                    ("inference", runtime["reid_inference"]),
                    ("postprocess", runtime["reid_postprocess"]),
                ),
                runtime["reid_total"],
            )
        )
    return rows


def _compact_timing_row(
    label: str,
    breakdown: tuple[tuple[str, float], ...],
    total_ms: float,
    timed_frames: int,
    *,
    denominator: str,
    strong: bool = False,
) -> Text:
    average = _average_ms(total_ms, timed_frames)
    text = Text()
    text.append(label, style=ui.STYLE_TEXT_STRONG if strong else ui.STYLE_ACCENT)
    text.append("  ")
    text.append(_format_ms(total_ms, decimals=1, unit=True), style=ui.STYLE_TEXT_STRONG if strong else ui.STYLE_TEXT)
    text.append(" total", style=ui.STYLE_MUTED)
    text.append("  ·  ", style=ui.STYLE_SUBTLE)
    text.append(_format_ms(average, decimals=2, unit=True), style=ui.STYLE_TEXT_STRONG if strong else ui.STYLE_TEXT)
    text.append(f"/{denominator}", style=ui.STYLE_MUTED)
    text.append("  ·  ", style=ui.STYLE_SUBTLE)
    text.append(_format_fps(average), style=ui.STYLE_TEXT_STRONG if strong else ui.STYLE_TEXT)
    text.append(" FPS", style=ui.STYLE_MUTED)
    if breakdown:
        text.append("\n  ")
        for index, (stage, value) in enumerate(breakdown):
            if index:
                text.append("  ·  ", style=ui.STYLE_SUBTLE)
            text.append(f"{stage} ", style=ui.STYLE_MUTED)
            text.append(_format_ms(value, decimals=1, unit=True), style=ui.STYLE_TEXT)
    return text


def _build_perception_table(
    runtime: Mapping[str, float],
    timed_frames: int,
    *,
    include_segmentor: bool,
    include_reid: bool,
    compact: bool,
    denominator: str,
) -> Table:
    rows = _runtime_component_rows(
        runtime,
        include_segmentor=include_segmentor,
        include_reid=include_reid,
    )
    if compact:
        table = Table.grid(expand=True, padding=(0, 0))
        table.add_column(overflow="fold")
        for label, breakdown, total_ms in rows:
            table.add_row(
                _compact_timing_row(
                    label,
                    breakdown,
                    total_ms,
                    timed_frames,
                    denominator=denominator,
                )
            )
        return table

    table = Table(
        expand=True,
        box=None,
        show_header=True,
        header_style=ui.STYLE_TABLE_HEADER,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Component", style=ui.STYLE_ACCENT, no_wrap=True, ratio=1)
    table.add_column("Pre (ms)", justify="right", no_wrap=True)
    table.add_column("Infer (ms)", justify="right", no_wrap=True)
    table.add_column("Post (ms)", justify="right", no_wrap=True)
    table.add_column("Total (ms)", justify="right", no_wrap=True)
    table.add_column(f"Avg/{denominator} (ms)", justify="right", no_wrap=True)
    table.add_column("FPS", justify="right", no_wrap=True)
    for label, breakdown, total_ms in rows:
        average = _average_ms(total_ms, timed_frames)
        phases = [value for _, value in breakdown]
        table.add_row(
            label,
            _format_ms(phases[0], decimals=1),
            _format_ms(phases[1], decimals=1),
            _format_ms(phases[2], decimals=1),
            _format_ms(total_ms, decimals=1),
            _format_ms(average, decimals=2),
            _format_fps(average),
        )
    return table


def _runtime_engine_rows(
    runtime: Mapping[str, float],
) -> list[tuple[str, tuple[tuple[str, float], ...], float, bool]]:
    return [
        ("Source", (("acquisition", runtime["source_acquisition"]),), runtime["source_acquisition"], False),
        (
            "Pipeline",
            (("enrichment", runtime["enrichment"]), ("validation", runtime["validation"])),
            runtime["enrichment"] + runtime["validation"],
            False,
        ),
        (
            "Tracker",
            (("association", runtime["tracker_association"]), ("update", runtime["tracker_update"])),
            runtime["tracker_total"],
            False,
        ),
        (
            "Output",
            (("rendering", runtime["rendering"]), ("sink I/O", runtime["sink_io"])),
            runtime["rendering"] + runtime["sink_io"],
            False,
        ),
        ("Other overhead", (), runtime["other_overhead"], False),
        ("Overall", (), runtime["overall"], True),
    ]


def _breakdown_text(values: tuple[tuple[str, float], ...]) -> Text:
    text = Text()
    for index, (stage, value) in enumerate(values):
        if index:
            text.append("  ·  ", style=ui.STYLE_SUBTLE)
        text.append(f"{stage} ", style=ui.STYLE_MUTED)
        text.append(_format_ms(value, decimals=1), style=ui.STYLE_TEXT)
    return text


def _build_engine_table(
    runtime: Mapping[str, float],
    timed_frames: int,
    *,
    compact: bool,
    denominator: str,
) -> Table:
    rows = _runtime_engine_rows(runtime)
    if compact:
        table = Table.grid(expand=True, padding=(0, 0))
        table.add_column(overflow="fold")
        for label, breakdown, total_ms, strong in rows:
            table.add_row(
                _compact_timing_row(
                    label,
                    breakdown,
                    total_ms,
                    timed_frames,
                    denominator=denominator,
                    strong=strong,
                )
            )
        return table

    table = Table(
        expand=True,
        box=None,
        show_header=True,
        header_style=ui.STYLE_TABLE_HEADER,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column("Stage", style=ui.STYLE_ACCENT, no_wrap=True)
    table.add_column("Breakdown (ms)", ratio=1)
    table.add_column("Total (ms)", justify="right", no_wrap=True)
    table.add_column(f"Avg/{denominator} (ms)", justify="right", no_wrap=True)
    table.add_column("FPS", justify="right", no_wrap=True)
    for label, breakdown, total_ms, strong in rows:
        average = _average_ms(total_ms, timed_frames)
        style = ui.STYLE_TEXT_STRONG if strong else None
        table.add_row(
            label,
            _breakdown_text(breakdown),
            _format_ms(total_ms, decimals=1),
            _format_ms(average, decimals=2),
            _format_fps(average),
            style=style,
            end_section=label == "Other overhead",
        )
    return table


def _build_output_table(run: Any) -> Table | None:
    outputs = [
        (label, getattr(run, attribute, None))
        for label, attribute in (
            ("Video", "video_path"),
            ("MOT text", "mot_path"),
            ("JSON Lines", "json_path"),
        )
        if getattr(run, attribute, None) is not None
    ]
    if not outputs:
        return None
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style=ui.STYLE_ACCENT, no_wrap=True)
    table.add_column(style=ui.STYLE_TEXT, ratio=1, overflow="fold")
    for label, path in outputs:
        rendered_path = Text(str(path), style=ui.STYLE_TEXT, overflow="fold")
        try:
            rendered_path.stylize(f"link {Path(path).expanduser().resolve().as_uri()}")
        except (OSError, TypeError, ValueError):
            pass
        table.add_row(label, rendered_path)
    return table


def _build_track_result(run: Any, *, compact: bool) -> Group:
    summary = run.summary
    completed_frames = _summary_counter(summary, "frames")
    timed_frames = _timed_frame_count(summary, completed_frames)
    denominator = "attempt" if timed_frames != completed_frames else "frame"
    startup = normalize_track_startup_timings_ms(_summary_mapping(summary, "startup_timings_ms", "setup_timings_ms"))
    raw_runtime = _summary_mapping(
        summary,
        "stage_timings_ms",
        "runtime_timings_ms",
        "timing_totals_ms",
        "timings_ms",
    )
    runtime = normalize_track_runtime_timings_ms(
        raw_runtime,
        elapsed_ms=float(getattr(summary, "elapsed_ms", 0.0) or 0.0),
    )
    include_segmentor = startup["segmentor_load"] > 0.0 or runtime["segmentor_total"] > 0.0
    include_reid = startup["reid_encoder_load"] > 0.0 or runtime["reid_total"] > 0.0

    sections: list[RenderableType] = [
        _build_status_summary(summary, runtime, timed_frames=timed_frames, compact=compact),
        _build_startup_section(startup, compact=compact),
        Rule(Text("Runtime", style=ui.STYLE_TITLE), style=ui.STYLE_RULE),
        Text("Perception", style=ui.STYLE_ACCENT),
        _build_perception_table(
            runtime,
            timed_frames,
            include_segmentor=include_segmentor,
            include_reid=include_reid,
            compact=compact,
            denominator=denominator,
        ),
        Text("Engine stages", style=ui.STYLE_ACCENT),
        _build_engine_table(
            runtime,
            timed_frames,
            compact=compact,
            denominator=denominator,
        ),
    ]
    if timed_frames != completed_frames:
        sections.append(
            Text(
                f"Timing averages and stage FPS include {timed_frames:,} attempted "
                f"frame{'s' if timed_frames != 1 else ''} ({completed_frames:,} completed).",
                style=ui.STYLE_MUTED,
            )
        )
    output_table = _build_output_table(run)
    if output_table is not None:
        sections.extend((Rule(Text("Outputs", style=ui.STYLE_TITLE), style=ui.STYLE_RULE), output_table))
    return Group(*sections)


class _ResponsiveTrackResult:
    """Select a legible final-report layout from the actual available width."""

    _WIDE_MIN_WIDTH = 120

    def __init__(self, run: Any) -> None:
        self.run = run

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console
        yield _build_track_result(self.run, compact=options.max_width < self._WIDE_MIN_WIDTH)


class TrackWorkflowReporter(RichWorkflowReporter):
    title = "Tracking"
    SETUP = 0
    RUN = 1
    steps = TRACK_STEPS

    def fields(self) -> list[tuple[str, object]]:
        return _build_track_workflow_fields(self.args)

    def result(self, run: Any) -> RenderableType:
        """Render complete or partial counters and engine-owned timing metadata."""

        return _ResponsiveTrackResult(run)


def log_track_pipeline_intro(args: Any) -> ui.WorkflowProgress:
    return TrackWorkflowReporter(args).create()


__all__ = [
    "TRACK_SETUP_STEP",
    "TRACK_RUN_STEP",
    "TrackWorkflowReporter",
    "log_track_pipeline_intro",
]
