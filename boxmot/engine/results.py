from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from boxmot.data import iter_source
from boxmot.detectors.base import Detections
from boxmot.utils import logger as LOGGER
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format, write_mot_results
from boxmot.utils.rich.ui import print_text


try:
    from ultralytics.utils.plotting import colors
except ImportError:
    colors = None


Drawer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _track_color(track_id: int) -> tuple[int, int, int]:
    if colors is not None:
        color = colors(track_id, True)
        return int(color[0]), int(color[1]), int(color[2])
    base = (int(track_id) * 123457) % 255
    return int(base), int((base * 3) % 255), int((base * 7) % 255)


def _is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isdigit() or "://" in source
    return False


class Tracks:
    def __init__(
        self,
        frame_idx: int,
        frame: np.ndarray,
        tracks: np.ndarray,
        detections: np.ndarray | None,
        source_path: str,
        get_drawer: Callable[[], Drawer | None],
        stop_session: Callable[[str | None], None] | None = None,
    ) -> None:
        self.frame_idx = int(frame_idx)
        self.frame = frame
        self.tracks = self._as_2d_array(tracks)
        self.detections = None if detections is None else self._as_2d_array(detections)
        self.source_path = source_path
        self._get_drawer = get_drawer
        self._stop_session = stop_session

    @staticmethod
    def _as_2d_array(values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            cols = arr.shape[1] if arr.ndim == 2 else 0
            return np.empty((0, cols), dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @property
    def num_tracks(self) -> int:
        return int(self.tracks.shape[0])

    def _default_draw(self, frame: np.ndarray) -> np.ndarray:
        drawn = frame.copy()
        if self.tracks.size == 0:
            return drawn

        is_obb = self.tracks.shape[1] >= 9
        for track in self.tracks:
            if is_obb:
                cx, cy, width, height, angle = track[:5]
                track_id = int(track[5])
                conf = float(track[6])
                rect = ((float(cx), float(cy)), (max(float(width), 1.0), max(float(height), 1.0)), float(np.degrees(angle)))
                corners = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(drawn, [corners], True, _track_color(track_id), 2)
                label_point = tuple(corners[0])
            else:
                x1, y1, x2, y2 = track[:4].round().astype(int)
                track_id = int(track[4])
                conf = float(track[5])
                cv2.rectangle(drawn, (x1, y1), (x2, y2), _track_color(track_id), 2)
                label_point = (x1, max(0, y1 - 6))

            cv2.putText(
                drawn,
                f"{track_id} {conf:.2f}",
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _track_color(track_id),
                1,
                cv2.LINE_AA,
            )

        return drawn

    def render(self) -> np.ndarray:
        drawer = self._get_drawer()
        if drawer is not None:
            return drawer(self.frame.copy(), self.tracks)
        return self._default_draw(self.frame)

    def show(self, window_name: str = "Tracking") -> bool:
        cv2.imshow(window_name, self.render())
        key = cv2.waitKey(1) & 0xFF
        should_continue = key not in (ord("q"), 27)
        if not should_continue and self._stop_session is not None:
            self._stop_session("Tracking stopped by user.")
        return should_continue

    def to_mot(self) -> np.ndarray:
        if self.tracks.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.tracks.shape[1] >= 9:
            return convert_to_mmot_obb_format(self.tracks, self.frame_idx)
        return convert_to_mot_format(self.tracks, self.frame_idx)

    def __str__(self) -> str:
        rows = self.to_mot()
        if rows.size == 0:
            return ""
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        buffer = io.StringIO()
        if rows.shape[1] == 9:
            np.savetxt(buffer, rows, fmt="%d,%d,%d,%d,%d,%d,%.6f,%d,%d")
        else:
            np.savetxt(buffer, rows, fmt="%g", delimiter=",")
        return buffer.getvalue()


class Results:
    def __init__(
        self,
        source,
        detector: Any,
        reid: Any,
        tracker: Any,
        verbose: bool = True,
        drawer: Drawer | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if detector is None:
            raise ValueError("A detector instance is required.")
        if tracker is None:
            raise ValueError("A tracker instance is required.")

        self.source = source
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.verbose = bool(verbose)
        self.drawer = drawer
        self._progress_callback = progress_callback
        self._generator: Iterator[Tracks] | None = None
        self._cache: list[Tracks] = []
        self._cache_results = not _is_live_source(source)
        self._exhausted = False
        self._interrupted = False
        self._track_ids_seen: set[int] = set()
        self.totals = {
            "det": 0.0,
            "reid": 0.0,
            "track": 0.0,
            "total": 0.0,
            "frames": 0,
            "detections": 0,
            "tracks": 0,
        }

    def __iter__(self):
        if self._exhausted:
            return iter(self._cache)
        if self._generator is None:
            self._generator = self._process()
        return self

    def __next__(self) -> Tracks:
        if self._generator is None:
            self._generator = self._process()
        try:
            result = next(self._generator)
        except StopIteration:
            self._exhausted = True
            raise
        if self._cache_results:
            self._cache.append(result)
        return result

    @staticmethod
    def _as_2d_array(values: Any, empty_cols: int = 0) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            cols = arr.shape[1] if arr.ndim == 2 else empty_cols
            return np.empty((0, cols), dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def _extract_detections(output: Any) -> np.ndarray:
        if isinstance(output, (list, tuple)) and len(output) == 1:
            output = output[0]
        if isinstance(output, Detections):
            cols = output.dets.shape[1] if output.dets.ndim == 2 else (7 if output.is_obb else 6)
            return Results._as_2d_array(output.dets, empty_cols=cols)
        if hasattr(output, "dets"):
            dets = getattr(output, "dets")
            cols = dets.shape[1] if isinstance(dets, np.ndarray) and dets.ndim == 2 else 6
            return Results._as_2d_array(dets, empty_cols=cols)
        if output is None:
            return np.empty((0, 6), dtype=np.float32)
        return Results._as_2d_array(output, empty_cols=6)

    def _iter_frames(self):
        source = self.source
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.is_dir() and (source_path / "img1").is_dir():
                source = source_path / "img1"
        yield from iter_source(source)

    def _log_frame_timings(self, frame_idx: int, det_ms: float, reid_ms: float, track_ms: float) -> None:
        total_ms = det_ms + reid_ms + track_ms
        if self.reid is None and reid_ms <= 0.0:
            message = f"Frame {frame_idx} | Det: {det_ms:.1f}ms | Track: {track_ms:.1f}ms | Total: {total_ms:.1f}ms"
            if self._progress_callback is not None:
                self._progress_callback(message)
            else:
                LOGGER.info(message)
            return
        message = (
            f"Frame {frame_idx} | Det: {det_ms:.1f}ms | ReID: {reid_ms:.1f}ms | "
            f"Track: {track_ms:.1f}ms | Total: {total_ms:.1f}ms"
        )
        if self._progress_callback is not None:
            self._progress_callback(message)
        else:
            LOGGER.info(message)

    def _tracker_reid_time_ms(self) -> float:
        getter = getattr(self.tracker, "get_last_reid_time_ms", None)
        value = getter() if callable(getter) else getattr(self.tracker, "last_reid_time_ms", 0.0)
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _log_summary(self) -> None:
        self.print_summary()

    def _run_reid(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray | None:
        if self.reid is None:
            return None
        try:
            return self.reid(frame, boxes=dets)
        except TypeError:
            return self.reid(frame, dets)

    def _run_tracker(self, dets: np.ndarray, frame: np.ndarray, features: np.ndarray | None) -> np.ndarray:
        if features is None:
            return self._as_2d_array(self.tracker.update(dets, frame), empty_cols=8)
        try:
            tracks = self.tracker.update(dets, frame, features)
        except TypeError:
            tracks = self.tracker.update(dets, frame)
        return self._as_2d_array(tracks, empty_cols=8)

    @staticmethod
    def _extract_track_ids(tracks: np.ndarray) -> set[int]:
        arr = np.asarray(tracks, dtype=np.float32)
        if arr.size == 0 or arr.ndim != 2:
            return set()
        if arr.shape[1] >= 9:
            return {int(track_id) for track_id in arr[:, 5].tolist()}
        if arr.shape[1] >= 8:
            return {int(track_id) for track_id in arr[:, 4].tolist()}
        return set()

    def _summary_snapshot(self) -> dict[str, Any]:
        frames = int(self.totals["frames"])
        avg_total = (self.totals["total"] / frames) if frames else 0.0
        return {
            "source": str(self.source),
            "frames": frames,
            "detections": int(self.totals["detections"]),
            "tracks": int(self.totals["tracks"]),
            "unique_tracks": len(self._track_ids_seen),
            "timings_ms": {
                "det": float(self.totals["det"]),
                "reid": float(self.totals["reid"]),
                "track": float(self.totals["track"]),
                "total": float(self.totals["total"]),
                "avg_total": float(avg_total),
            },
        }

    def stop(self, reason: str | None = None) -> None:
        if self._exhausted:
            return

        self._interrupted = True
        if reason:
            if self._progress_callback is not None:
                self._progress_callback(reason)
            else:
                LOGGER.info(reason)

        generator = self._generator
        self._generator = None
        if generator is not None:
            generator.close()
        else:
            self._exhausted = True

    def format_summary(self) -> str:
        summary = self.summary()
        timings = summary["timings_ms"]
        frames = max(int(summary["frames"]), 1)
        width = 86

        def _fps(total_ms: float) -> float:
            avg_ms = float(total_ms) / frames if frames else 0.0
            return (1000.0 / avg_ms) if avg_ms else 0.0

        lines = [
            "=" * width,
            f"{'TRACKING SUMMARY':^{width}}",
            "=" * width,
            f"Source:      {summary['source']}",
            f"Frames:      {summary['frames']}",
            f"Detections:  {summary['detections']}",
            f"Track rows:  {summary['tracks']}",
            f"Unique IDs:  {summary.get('unique_tracks', 0)}",
            "-" * width,
            f"{'Component':<14} {'Total (ms)':>12} {'Avg (ms)':>12} {'FPS':>10}",
            "-" * width,
            f"{'Detection':<14} {timings['det']:>12.1f} {(timings['det'] / frames if frames else 0.0):>12.2f} {_fps(timings['det']):>10.1f}",
            f"{'ReID':<14} {timings['reid']:>12.1f} {(timings['reid'] / frames if frames else 0.0):>12.2f} {_fps(timings['reid']):>10.1f}",
            f"{'Tracking':<14} {timings['track']:>12.1f} {(timings['track'] / frames if frames else 0.0):>12.2f} {_fps(timings['track']):>10.1f}",
            "-" * width,
            f"{'Total':<14} {timings['total']:>12.1f} {timings['avg_total']:>12.2f} {_fps(timings['total']):>10.1f}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        frames = int(self.totals["frames"])
        if frames == 0:
            return
        print_text(self.format_summary())

    def _process(self):
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()

        try:
            for frame_idx, (path, frame) in enumerate(self._iter_frames(), start=1):
                det_started = time.perf_counter()
                detector_output = self.detector(frame)
                dets = self._extract_detections(detector_output)
                det_ms = (time.perf_counter() - det_started) * 1000

                reid_ms = 0.0
                if self.reid is not None:
                    reid_started = time.perf_counter()
                    features = self._run_reid(frame, dets)
                    reid_ms = (time.perf_counter() - reid_started) * 1000
                else:
                    features = None

                track_started = time.perf_counter()
                tracks = self._run_tracker(dets, frame, features)
                track_ms = (time.perf_counter() - track_started) * 1000
                if self.reid is None:
                    tracker_reid_ms = min(self._tracker_reid_time_ms(), track_ms)
                    reid_ms += tracker_reid_ms
                    track_ms = max(track_ms - tracker_reid_ms, 0.0)

                total_ms = det_ms + reid_ms + track_ms
                self.totals["det"] += det_ms
                self.totals["reid"] += reid_ms
                self.totals["track"] += track_ms
                self.totals["total"] += total_ms
                self.totals["frames"] += 1
                self.totals["detections"] += int(dets.shape[0])
                self.totals["tracks"] += int(tracks.shape[0])
                self._track_ids_seen.update(self._extract_track_ids(tracks))

                if self.verbose or self._progress_callback is not None:
                    self._log_frame_timings(frame_idx, det_ms, reid_ms, track_ms)

                yield Tracks(
                    frame_idx=frame_idx,
                    frame=frame,
                    tracks=tracks,
                    detections=dets,
                    source_path=path,
                    get_drawer=lambda: self.drawer,
                    stop_session=self.stop,
                )
        except KeyboardInterrupt:
            self._interrupted = True
            if self._progress_callback is not None:
                self._progress_callback("Tracking interrupted by user.")
            else:
                LOGGER.info("Tracking interrupted by user.")
            return
        finally:
            self._exhausted = True
            if self.verbose:
                self._log_summary()

    def materialize(self) -> list[Tracks]:
        while not self._exhausted:
            try:
                next(self)
            except StopIteration:
                break
        return self._cache

    def save(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        for track_result in self.materialize():
            write_mot_results(path, track_result.to_mot())
        return path

    def summary(self) -> dict[str, Any]:
        if not self._exhausted and not self._interrupted and not _is_live_source(self.source):
            self.materialize()
        return self._summary_snapshot()

    def show(self) -> None:
        for track_result in self:
            if not track_result.show():
                break
        cv2.destroyAllWindows()


def track(source, detector, reid=None, tracker=None, verbose: bool = True, drawer: Drawer | None = None) -> Results:
    return Results(source, detector, reid, tracker, verbose=verbose, drawer=drawer)
