from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from boxmot.engine.tracking.results import Results


def _is_leaf_source(path: Path) -> bool:
    from boxmot.data import IMAGE_EXTS, VIDEO_EXTS

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


def _coerce_results(
    data: Any,
    *,
    detector=None,
    reid=None,
    tracker=None,
    verbose: bool = False,
    track_fn=None,
) -> list[Results]:
    from boxmot.engine.tracking.results import Results

    if isinstance(data, Results):
        return [data]

    if isinstance(data, (list, tuple)) and all(isinstance(item, Results) for item in data):
        return list(data)

    if detector is None or tracker is None:
        raise ValueError("Detector and tracker are required when evaluating raw sources.")
    if track_fn is None:
        raise ValueError("A tracking function is required when evaluating raw sources.")

    return [track_fn(source, detector, reid, tracker, verbose=verbose) for source in _expand_sources(data)]


def track(source, detector, reid=None, tracker=None, *, verbose: bool = True, drawer=None) -> Results:
    """Create a lazy streaming tracking result iterator."""
    if tracker is None:
        raise ValueError("A tracker instance is required.")

    from boxmot.engine.tracking.results import Results

    return Results(source, detector, reid, tracker, verbose=verbose, drawer=drawer)


def evaluate(
    data,
    detector=None,
    reid=None,
    tracker=None,
    *,
    metrics: bool = True,
    speed: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Aggregate run metrics over one or more tracking results."""
    runs = _coerce_results(
        data,
        detector=detector,
        reid=reid,
        tracker=tracker,
        verbose=verbose,
        track_fn=track,
    )
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


__all__ = ("evaluate", "track")
