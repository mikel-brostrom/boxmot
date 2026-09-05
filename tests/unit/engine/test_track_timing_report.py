from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from boxmot.engine.tracking.timing import (
    build_track_runtime_display_rows,
    normalize_track_runtime_timings_ms,
    normalize_track_startup_timings_ms,
)
from boxmot.engine.ui.core.ui import capture_renderable
from boxmot.engine.ui.reporters.track import TrackWorkflowReporter, _format_fps, _format_ms


def test_track_startup_timing_normalizes_partial_and_alias_values() -> None:
    timings = normalize_track_startup_timings_ms(
        {
            "detector_load": 10.0,
            "tracker_load": 4.0,
            "reid_load": 6.0,
            "segmentor_load": float("nan"),
            "pipeline_prepare": 2.0,
            "output_prepare": -3.0,
            "source_open": 5.0,
            "source_first_frame": 3.0,
        }
    )

    assert timings == {
        "detector_load": 10.0,
        "tracker_load": 4.0,
        "reid_encoder_load": 6.0,
        "segmentor_load": 0.0,
        "pipeline_prepare": 2.0,
        "output_prepare": 0.0,
        "source_open": 5.0,
        "source_first_frame": 3.0,
        "total": 30.0,
    }


def test_track_runtime_timing_derives_component_totals_and_other_overhead() -> None:
    timings = normalize_track_runtime_timings_ms(
        {
            "source_acquisition": 10.0,
            "detector_preprocess": 20.0,
            "detector_inference": 30.0,
            "detector_postprocess": 10.0,
            "detector_total": 50.0,
            "segmentor_preprocess": 5.0,
            "segmentor_inference": 10.0,
            "segmentor_postprocess": 5.0,
            "reid_preprocess": 8.0,
            "reid_process": 12.0,
            "reid_postprocess": 4.0,
            "enrichment": 3.0,
            "validation": 2.0,
            "tracker_association": 2.0,
            "tracker_update": 5.0,
            "rendering": 4.0,
            "sink_io": 2.0,
        },
        elapsed_ms=150.0,
    )

    assert timings["detector_total"] == 60.0
    assert timings["segmentor_total"] == 20.0
    assert timings["reid_inference"] == 12.0
    assert timings["reid_total"] == 24.0
    assert timings["tracker_total"] == 7.0
    assert timings["other_overhead"] == 18.0
    assert timings["overall"] == 150.0


def test_track_runtime_rows_report_every_phase_with_total_average_and_fps() -> None:
    rows = build_track_runtime_display_rows(
        {
            "detector_preprocess": 4.0,
            "detector_inference": 8.0,
            "detector_postprocess": 2.0,
            "segmentor_inference": 6.0,
            "reid_preprocess": 2.0,
            "reid_inference": 10.0,
            "reid_postprocess": 4.0,
            "tracker_association": 1.0,
            "tracker_update": 3.0,
        },
        2,
        elapsed_ms=50.0,
    )
    by_label = {str(row["label"]).strip(): row for row in rows if row["kind"] == "row"}

    assert by_label["preprocess"]["total"] in {2.0, 4.0}
    detector_inference = next(
        row for row in rows if row["kind"] == "row" and row["label"] == "  inference" and row["total"] == 8.0
    )
    assert detector_inference["avg"] == 4.0
    assert detector_inference["fps"] == 250.0
    assert "Segmentor total" in by_label
    assert "ReID total" in by_label
    assert "enrichment" in by_label
    assert "validation" in by_label
    assert by_label["association / update"]["total"] == 4.0
    assert "rendering" in by_label
    assert "sink I/O" in by_label
    assert "Other overhead" in by_label
    assert by_label["Overall"]["total"] == 50.0


def test_track_reporter_renders_detailed_final_timing_and_outputs() -> None:
    summary = SimpleNamespace(
        frames=2,
        detections=12,
        track_rows=9,
        unique_track_ids=4,
        sequences=1,
        elapsed_ms=100.0,
        interrupted=False,
        startup_timings_ms={
            "detector_load": 20.0,
            "tracker_load": 4.0,
            "reid_load": 8.0,
            "segmentor_load": 0.0,
            "pipeline_prepare": 2.0,
            "output_prepare": 3.0,
            "source_open": 5.0,
            "source_first_frame": 6.0,
        },
        stage_timings_ms={
            "source_acquisition": 4.0,
            "detector_preprocess": 6.0,
            "detector_inference": 20.0,
            "detector_postprocess": 4.0,
            "reid_preprocess": 6.0,
            "reid_inference": 12.0,
            "reid_postprocess": 2.0,
            "enrichment": 3.0,
            "validation": 2.0,
            "tracker_association": 1.0,
            "tracker_update": 4.0,
            "rendering": 8.0,
            "sink_io": 10.0,
        },
    )
    run = SimpleNamespace(
        summary=summary,
        video_path=Path("runs/track/tracks.mp4"),
        mot_path=None,
        json_path=None,
    )

    rendered = capture_renderable(TrackWorkflowReporter(SimpleNamespace()).result(run), width=160)

    for expected in (
        "Tracking complete",
        "2 frames",
        "12 detections",
        "9 track rows",
        "4 unique IDs",
        "1 sequence",
        "Startup",
        "Detector load",
        "Tracker load",
        "ReID encoder load",
        "Segmentor load",
        "Pipeline preparation",
        "Output preparation",
        "Source open",
        "First source frame",
        "also counted in Source acquisition runtime",
        "Runtime",
        "Perception",
        "Component",
        "Pre (ms)",
        "Infer (ms)",
        "Post (ms)",
        "Source",
        "Detector",
        "ReID encoder",
        "Engine stages",
        "acquisition",
        "enrichment",
        "validation",
        "association",
        "update",
        "rendering",
        "sink I/O",
        "Other overhead",
        "Overall",
        "Total (ms)",
        "Avg/frame (ms)",
        "FPS",
        "Outputs",
        "runs/track/tracks.mp4",
    ):
        assert expected in rendered
    assert "Counters" not in rendered
    assert "Segmentor\n" not in rendered


@pytest.mark.parametrize("interrupted_field", ("interrupted", "stopped_by_user"))
def test_track_reporter_renders_partial_zero_frame_shutdown(interrupted_field: str) -> None:
    summary_values = {
        "frames": 0,
        "detections": 0,
        "track_rows": 0,
        "unique_track_ids": 0,
        "sequences": 1,
        "elapsed_ms": 42.0,
        "interrupted": False,
        "startup_timings_ms": {"detector_load": 11.0, "reid_load": 2.0, "source_open": 3.0},
        "stage_timings_ms": {"detector_preprocess": 30.0},
        "timings": (SimpleNamespace(),),
    }
    summary_values[interrupted_field] = True
    run = SimpleNamespace(
        summary=SimpleNamespace(**summary_values),
        video_path=None,
        mot_path=None,
        json_path=None,
    )

    rendered = capture_renderable(TrackWorkflowReporter(SimpleNamespace()).result(run), width=140)

    assert "Tracking stopped by user" in rendered
    assert "0/1 frames completed" in rendered
    assert "Startup" in rendered
    assert "16.0 ms total" in rendered
    assert rendered.count("ReID encoder") >= 2
    assert "Overall" in rendered
    assert "42.0" in rendered
    assert "30.00" in rendered
    assert "42.0 ms runtime  •  — FPS" in rendered
    assert "Avg/attempt (ms)" in rendered
    assert "1 attempted frame (0 completed)" in rendered
    assert "Counters" not in rendered


def test_track_runtime_rows_omit_components_that_never_ran() -> None:
    rows = build_track_runtime_display_rows(
        {"detector_inference": 1.0, "tracker_association_update": 1.0},
        1,
    )
    groups = [row["label"] for row in rows if row["kind"] == "group"]

    assert "Segmentor" not in groups
    assert "ReID encoder" not in groups


def test_track_timing_formatters_distinguish_zero_and_positive_tiny_values() -> None:
    assert _format_ms(0.0, decimals=1, unit=True) == "—"
    assert _format_ms(0.01, decimals=1, unit=True) == "<0.1 ms"
    assert _format_ms(0.001, decimals=2, unit=True) == "<0.01 ms"
    assert _format_fps(0.0) == "—"
    assert _format_fps(100_000.0) == "<0.1"
    assert _format_fps(0.5) == "2.0k"
    assert _format_fps(0.001) == "1.0M"


@pytest.mark.parametrize("width", (80, 112, 140))
def test_track_reporter_is_responsive_without_truncating_timing_categories(width: int) -> None:
    summary = SimpleNamespace(
        frames=2,
        detections=12,
        track_rows=9,
        unique_track_ids=4,
        sequences=1,
        elapsed_ms=100.0,
        interrupted=False,
        timings=(SimpleNamespace(), SimpleNamespace()),
        startup_timings_ms={
            "detector_load": 20.0,
            "tracker_load": 4.0,
            "reid_load": 8.0,
            "segmentor_load": 2.0,
            "pipeline_prepare": 0.01,
            "output_prepare": 3.0,
            "source_open": 5.0,
            "source_first_frame": 6.0,
        },
        stage_timings_ms={
            "source_acquisition": 4.0,
            "detector_preprocess": 6.0,
            "detector_inference": 20.0,
            "detector_postprocess": 4.0,
            "segmentor_preprocess": 0.01,
            "segmentor_inference": 6.0,
            "segmentor_postprocess": 0.02,
            "reid_preprocess": 6.0,
            "reid_inference": 12.0,
            "reid_postprocess": 2.0,
            "enrichment": 3.0,
            "validation": 2.0,
            "tracker_association": 0.01,
            "tracker_update": 4.0,
            "rendering": 8.0,
            "sink_io": 10.0,
        },
    )
    run = SimpleNamespace(
        summary=summary,
        video_path=Path("runs/track/a/very/long/output/directory/that/must/fold/without/being/truncated/tracks.mp4"),
        mot_path=None,
        json_path=None,
    )

    rendered = capture_renderable(TrackWorkflowReporter(SimpleNamespace()).result(run), width=width)

    assert max(map(len, rendered.splitlines())) <= width
    assert "…" not in rendered
    assert "tracks.mp4" in rendered
    for category in (
        "Detector",
        "Segmentor",
        "ReID encoder",
        "Source",
        "acquisition",
        "Pipeline",
        "enrichment",
        "validation",
        "Tracker",
        "association",
        "update",
        "Output",
        "rendering",
        "sink I/O",
        "Other overhead",
        "Overall",
    ):
        assert category in rendered
    assert "<0.1" in rendered
    if width < 120:
        assert "Component" not in rendered
        assert "preprocess" in rendered
        assert "inference" in rendered
        assert "postprocess" in rendered
    else:
        assert "Component" in rendered
        assert "Pre (ms)" in rendered
        assert "Infer (ms)" in rendered
        assert "Post (ms)" in rendered
