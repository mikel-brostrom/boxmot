from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from boxmot.data import iter_source
from boxmot.detectors.base import Detections
from boxmot.trackers.track_results import TrackResults
from boxmot.utils import logger as LOGGER
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format, write_mot_results
from boxmot.utils.rich.ui import print_text
from boxmot.utils.timing import build_timing_display_rows, derive_timing_breakdown


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


class FrameResult:
    """Per-frame tracking result container.

    Bundles the source frame with its TrackResults (returned by the tracker)
    and adds visualization and frame-aware export.  Track data is accessed
    directly via ``self.tracks`` (a TrackResults instance with .id, .conf,
    .cls, .xyxy, .xywha, etc.).

    Attributes:
        frame_idx (int): 1-based frame index.
        frame (np.ndarray): The source frame (HxWxC BGR).
        tracks (TrackResults): Track output array with named accessors.
        detections (np.ndarray | None): Raw detector output for this frame.
        source_path (str): Path of the source file/stream.

    Methods:
        plot: Render tracks on the frame and return the annotated image.
        show: Display the annotated frame in a window.
        save: Save the annotated frame to disk.
        save_txt: Append tracks in MOT format to a text file.
        save_csv: Append tracks in CSV format to a file.
        summary: Return tracks as a list of dicts.
        to_json: Return tracks as a JSON string.
        to_csv: Return tracks as a CSV string.
        verbose: Return a human-readable summary string.
    """

    def __init__(
        self,
        frame_idx: int,
        frame: np.ndarray,
        tracks: TrackResults | np.ndarray,
        detections: np.ndarray | None,
        source_path: str,
        get_drawer: Callable[[], Drawer | None],
        stop_session: Callable[[str | None], None] | None = None,
    ) -> None:
        self.frame_idx = int(frame_idx)
        self.frame = frame
        self.tracks = tracks if isinstance(tracks, TrackResults) else TrackResults(tracks)
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

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        """Number of active tracks in this frame."""
        return int(self.tracks.shape[0])

    def __len__(self) -> int:
        return self.num_tracks

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _default_draw(self, frame: np.ndarray) -> np.ndarray:
        drawn = frame.copy()
        if self.tracks.size == 0:
            return drawn

        for i in range(len(self.tracks)):
            track_id = int(self.tracks.id[i])
            conf = float(self.tracks.conf[i])
            if self.tracks.is_obb:
                cx, cy, width, height, angle = self.tracks.xywha[i]
                rect = ((float(cx), float(cy)), (max(float(width), 1.0), max(float(height), 1.0)), float(np.degrees(angle)))
                corners = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(drawn, [corners], True, _track_color(track_id), 2)
                label_point = tuple(corners[0])
            else:
                x1, y1, x2, y2 = self.tracks.xyxy[i].round().astype(int)
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

    def plot(self) -> np.ndarray:
        """Plot tracks on the frame and return the annotated image."""
        drawer = self._get_drawer()
        if drawer is not None:
            return drawer(self.frame.copy(), self.tracks)
        return self._default_draw(self.frame)

    def render(self) -> np.ndarray:
        """Alias for plot()."""
        return self.plot()

    def show(self, window_name: str = "Tracking") -> bool:
        """Display the annotated frame. Returns False if user pressed 'q'."""
        cv2.imshow(window_name, self.plot())
        key = cv2.waitKey(1) & 0xFF
        should_continue = key not in (ord("q"), 27)
        if not should_continue and self._stop_session is not None:
            self._stop_session("Tracking stopped by user.")
        return should_continue

    def save(self, filename: str | Path | None = None) -> str:
        """Save annotated frame to disk.

        Args:
            filename: Output path. Defaults to 'result_<frame_idx>.jpg'.

        Returns:
            str: Path where the image was saved.
        """
        if filename is None:
            filename = f"result_{self.frame_idx:06d}.jpg"
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.plot())
        return str(path)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_mot(self) -> np.ndarray:
        """Convert tracks to MOT challenge format array."""
        if self.tracks.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.tracks.is_obb:
            return convert_to_mmot_obb_format(self.tracks, self.frame_idx)
        return convert_to_mot_format(self.tracks, self.frame_idx)

    def save_txt(self, path: str | Path) -> None:
        """Append tracks in MOT challenge format to a text file."""
        self.tracks.save_mot(path, frame_id=self.frame_idx)

    def save_csv(self, path: str | Path, header: bool = True) -> None:
        """Append tracks in CSV format to a file."""
        self.tracks.save_csv(path, frame_id=self.frame_idx, header=header)

    def to_csv(self) -> str:
        """Return tracks as a CSV-formatted string."""
        return self.tracks.to_csv(frame_id=self.frame_idx)

    def to_json(self, indent: int | None = None) -> str:
        """Return tracks as a JSON string."""
        return self.tracks.to_json(indent=indent)

    def summary(self) -> list[dict[str, Any]]:
        """Return tracks as a list of dictionaries."""
        return self.tracks.summary()

    def verbose(self) -> str:
        """Return a human-readable summary string for this frame."""
        n = self.num_tracks
        if n == 0:
            return f"Frame {self.frame_idx}: (no tracks)"
        ids = ", ".join(str(i) for i in self.tracks.id[:5])
        suffix = f", ... ({n} total)" if n > 5 else ""
        return f"Frame {self.frame_idx}: {n} tracks [IDs: {ids}{suffix}]"

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

    def __repr__(self) -> str:
        return f"Tracks(frame={self.frame_idx}, n={self.num_tracks}, obb={self.tracks.is_obb})"


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
        self._generator: Iterator[FrameResult] | None = None
        self._cache: list[FrameResult] = []
        self._cache_results = not _is_live_source(source)
        self._exhausted = False
        self._interrupted = False
        self._track_ids_seen: set[int] = set()
        self.totals = {
            "det": 0.0,
            "detector_preprocess": 0.0,
            "detector_process": 0.0,
            "detector_postprocess": 0.0,
            "reid": 0.0,
            "reid_preprocess": 0.0,
            "reid_process": 0.0,
            "reid_postprocess": 0.0,
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

    def __next__(self) -> FrameResult:
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

    def _tracker_reid_phase_breakdown(self) -> dict[str, float]:
        breakdown: dict[str, float] = {"preprocess": 0.0, "process": 0.0, "postprocess": 0.0}
        for phase in breakdown:
            getter = getattr(self.tracker, f"get_last_reid_{phase}_time_ms", None)
            value = getter() if callable(getter) else getattr(self.tracker, f"last_reid_{phase}_time_ms", 0.0)
            try:
                breakdown[phase] = max(float(value), 0.0)
            except (TypeError, ValueError):
                breakdown[phase] = 0.0
        return breakdown

    def _log_summary(self) -> None:
        self.print_summary()

    def _run_reid(self, frame: np.ndarray, dets: np.ndarray) -> np.ndarray | None:
        if self.reid is None:
            return None
        try:
            return self.reid(frame, boxes=dets)
        except TypeError:
            return self.reid(frame, dets)

    def _add_detector_phase_time(self, phase: str, time_ms: float) -> None:
        phase_key = f"detector_{phase}"
        elapsed_ms = float(time_ms or 0.0)
        self.totals[phase_key] += elapsed_ms
        self.totals["det"] += elapsed_ms

    def _add_reid_phase_time(self, phase: str, time_ms: float) -> None:
        phase_key = f"reid_{phase}"
        elapsed_ms = float(time_ms or 0.0)
        self.totals[phase_key] += elapsed_ms
        self.totals["reid"] += elapsed_ms

    def _run_detector_timed(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if all(hasattr(self.detector, attr) for attr in ("preprocess", "process", "postprocess")):
            try:
                preprocess_started = time.perf_counter()
                preprocessed = self.detector.preprocess(frame)
                preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
                self._add_detector_phase_time("preprocess", preprocess_ms)

                process_started = time.perf_counter()
                raw_output = self.detector.process(preprocessed)
                process_ms = (time.perf_counter() - process_started) * 1000
                self._add_detector_phase_time("process", process_ms)

                postprocess_started = time.perf_counter()
                detector_output = self.detector.postprocess(raw_output)
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
                self._add_detector_phase_time("postprocess", postprocess_ms)

                dets = self._extract_detections(detector_output)
                return dets, preprocess_ms + process_ms + postprocess_ms
            except NotImplementedError:
                pass

        det_started = time.perf_counter()
        detector_output = self.detector(frame)
        det_ms = (time.perf_counter() - det_started) * 1000
        self._add_detector_phase_time("process", det_ms)
        dets = self._extract_detections(detector_output)
        return dets, det_ms

    def _run_reid_timed(self, frame: np.ndarray, dets: np.ndarray) -> tuple[np.ndarray | None, float]:
        if self.reid is None:
            return None, 0.0

        if all(hasattr(self.reid, attr) for attr in ("preprocess", "process", "postprocess")):
            try:
                preprocess_started = time.perf_counter()
                try:
                    payload = self.reid.preprocess(frame, boxes=dets)
                except TypeError:
                    payload = self.reid.preprocess(frame, dets)
                preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
                self._add_reid_phase_time("preprocess", preprocess_ms)

                process_started = time.perf_counter()
                try:
                    features = self.reid.process(payload, boxes=dets)
                except TypeError:
                    features = self.reid.process(payload, dets)
                process_ms = (time.perf_counter() - process_started) * 1000
                self._add_reid_phase_time("process", process_ms)

                postprocess_started = time.perf_counter()
                try:
                    features = self.reid.postprocess(features, boxes=dets)
                except TypeError:
                    features = self.reid.postprocess(features, dets)
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
                self._add_reid_phase_time("postprocess", postprocess_ms)
                return features, preprocess_ms + process_ms + postprocess_ms
            except NotImplementedError:
                pass

        reid_started = time.perf_counter()
        features = self._run_reid(frame, dets)
        reid_ms = (time.perf_counter() - reid_started) * 1000
        self._add_reid_phase_time("process", reid_ms)
        return features, reid_ms

    def _run_tracker(self, dets: np.ndarray, frame: np.ndarray, features: np.ndarray | None) -> TrackResults:
        if features is None:
            result = self.tracker.update(dets, frame)
        else:
            try:
                result = self.tracker.update(dets, frame, features)
            except TypeError:
                result = self.tracker.update(dets, frame)
        if isinstance(result, TrackResults):
            return result
        return TrackResults(result)

    @staticmethod
    def _extract_track_ids(tracks: TrackResults) -> set[int]:
        if tracks.size == 0 or tracks.ndim != 2:
            return set()
        return {int(tid) for tid in tracks.id.tolist()}

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
                "detector_preprocess": float(self.totals["detector_preprocess"]),
                "detector_process": float(self.totals["detector_process"]),
                "detector_postprocess": float(self.totals["detector_postprocess"]),
                "reid": float(self.totals["reid"]),
                "reid_preprocess": float(self.totals["reid_preprocess"]),
                "reid_process": float(self.totals["reid_process"]),
                "reid_postprocess": float(self.totals["reid_postprocess"]),
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
        breakdown = derive_timing_breakdown(timings, frames, total_time_ms=timings["total"])

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
            f"{'Stage':<20} {'Total (ms)':>12} {'Avg (ms)':>12} {'FPS':>10}",
            "-" * width,
        ]
        for entry in build_timing_display_rows(
            breakdown,
            frames,
            metadata=dict(getattr(self, "timing_metadata", {})),
            overall_avg_ms=float(timings["avg_total"]),
        ):
            if entry["kind"] == "group":
                lines.append(str(entry["label"]))
                continue
            if entry["kind"] == "note":
                lines.append(str(entry["label"]))
                continue
            lines.append(
                f"{str(entry['label']):<20} {float(entry['total']):>12.1f} {float(entry['avg']):>12.2f} {float(entry['fps']):>10.1f}"
            )
        lines.append("=" * width)
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
                dets, det_ms = self._run_detector_timed(frame)
                features, reid_ms = self._run_reid_timed(frame, dets)

                track_started = time.perf_counter()
                tracks = self._run_tracker(dets, frame, features)
                track_ms = (time.perf_counter() - track_started) * 1000
                if self.reid is None:
                    tracker_reid_ms = min(self._tracker_reid_time_ms(), track_ms)
                    reid_ms += tracker_reid_ms
                    phase_breakdown = self._tracker_reid_phase_breakdown()
                    if any(phase_breakdown.values()) and tracker_reid_ms > 0.0:
                        breakdown_total = sum(phase_breakdown.values())
                        scale = tracker_reid_ms / breakdown_total if breakdown_total > 0.0 else 1.0
                        for phase, value in phase_breakdown.items():
                            scaled = max(float(value) * scale, 0.0)
                            self._add_reid_phase_time(phase, scaled)
                    else:
                        self._add_reid_phase_time("process", tracker_reid_ms)
                    track_ms = max(track_ms - tracker_reid_ms, 0.0)

                total_ms = det_ms + reid_ms + track_ms
                self.totals["track"] += track_ms
                self.totals["total"] += total_ms
                self.totals["frames"] += 1
                self.totals["detections"] += int(dets.shape[0])
                self.totals["tracks"] += int(tracks.shape[0])
                self._track_ids_seen.update(self._extract_track_ids(tracks))

                if self.verbose or self._progress_callback is not None:
                    self._log_frame_timings(frame_idx, det_ms, reid_ms, track_ms)

                yield FrameResult(
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

    def materialize(self) -> list[FrameResult]:
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
