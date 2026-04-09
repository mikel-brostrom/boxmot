from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from boxmot.api import Boxmot
from boxmot.configs import get_mode_default
from boxmot.trackers.tracker_zoo import TRACKER_MAPPING, create_tracker, get_tracker_config
from boxmot.utils.mot_utils import convert_to_mmot_obb_format, convert_to_mot_format
from boxmot.utils.timing import TimingStats, wrap_tracker_reid
from boxmot.utils.torch_utils import select_device


def _primary_model_ref(value):
	if isinstance(value, (list, tuple)):
		return value[0] if value else None
	return value


def _is_live_source(source: Any) -> bool:
	if isinstance(source, int):
		return True
	if isinstance(source, str):
		return source.isdigit() or "://" in source
	return False


class TrackerRuntime:
	"""Wrap one tracker instance with timing and formatting helpers."""

	def __init__(self, tracker: Any, timing_stats: TimingStats | None = None) -> None:
		self.tracker = tracker
		self.timing_stats = timing_stats

	@classmethod
	def create(
		cls,
		tracking_method: str,
		reid_weights,
		device,
		half: bool,
		per_class: bool,
		evolve_param_dict: dict | None = None,
		target_id: int | None = None,
		timing_stats: TimingStats | None = None,
	) -> TrackerRuntime:
		"""Instantiate a tracker and wrap it in the runtime helper."""
		normalized_method = str(tracking_method).lower()
		if normalized_method not in TRACKER_MAPPING:
			available = ", ".join(sorted(TRACKER_MAPPING))
			raise ValueError(f"'{tracking_method}' is not supported. Supported ones are {available}")

		tracker = create_tracker(
			tracker_type=normalized_method,
			tracker_config=get_tracker_config(normalized_method),
			reid_weights=reid_weights,
			device=device,
			half=half,
			per_class=per_class,
			evolve_param_dict=evolve_param_dict,
		)
		if target_id is not None:
			tracker.target_id = target_id
		if timing_stats is not None:
			wrap_tracker_reid(tracker, timing_stats)
		return cls(tracker, timing_stats=timing_stats)

	@staticmethod
	def _ensure_2d_tracks(tracks: np.ndarray) -> np.ndarray:
		arr = np.asarray(tracks, dtype=np.float32)
		if arr.size == 0:
			if arr.ndim == 2:
				return arr
			return np.empty((0, 0), dtype=np.float32)
		if arr.ndim == 1:
			return arr.reshape(1, -1)
		return arr

	@staticmethod
	def format_for_mot(tracks: np.ndarray, frame_idx: int) -> np.ndarray:
		"""Convert one frame of tracker output to MOT or MMOT-OBB rows."""
		arr = TrackerRuntime._ensure_2d_tracks(tracks)
		if arr.size == 0:
			return np.empty((0, 0), dtype=np.float32)
		if arr.shape[1] >= 9:
			return convert_to_mmot_obb_format(arr, frame_idx)
		return convert_to_mot_format(arr, frame_idx)

	@property
	def names(self):
		return getattr(self.tracker, "names", None)

	@names.setter
	def names(self, value) -> None:
		setattr(self.tracker, "names", value)

	def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> tuple[np.ndarray, float]:
		"""Run one tracker update and return normalized tracks with elapsed milliseconds."""
		elapsed_ms = 0.0
		started = False
		if self.timing_stats is not None:
			self.timing_stats.reset_frame_reid()
			self.timing_stats.start_tracking()
			started = True
		else:
			start_time = time.perf_counter()

		try:
			if embs is None:
				tracks = self.tracker.update(dets, img)
			else:
				try:
					tracks = self.tracker.update(dets, img, embs)
				except TypeError:
					tracks = self.tracker.update(dets, img)
		finally:
			if started:
				self.timing_stats.end_tracking()
				elapsed_ms = self.timing_stats.get_last_track_time()
			else:
				elapsed_ms = (time.perf_counter() - start_time) * 1000

		return self._ensure_2d_tracks(tracks), elapsed_ms

	def plot_results(
		self,
		img: np.ndarray,
		show_trajectories: bool,
		*,
		thickness: int = 2,
		show_kf_preds: bool = False,
	) -> np.ndarray:
		"""Render tracker state onto a frame when supported by the wrapped tracker."""
		if hasattr(self.tracker, "plot_results"):
			return self.tracker.plot_results(
				img,
				show_trajectories,
				thickness=thickness,
				show_kf_preds=show_kf_preds,
			)
		return img

	def __getattr__(self, name: str):
		return getattr(self.tracker, name)


class TrackingSession:
	def __init__(self, args):
		self.args = args

	def _should_consume_result(self) -> bool:
		if getattr(self.args, "show", False):
			return False
		if getattr(self.args, "save", False) or getattr(self.args, "save_txt", False):
			return False
		return not _is_live_source(getattr(self.args, "source", None))

	def _resolve_output_stem(self) -> str:
		source = str(getattr(self.args, "source", ""))
		if source.isdigit():
			return f"camera_{source}"
		if "://" in source:
			parsed = urlparse(source)
			pieces = [parsed.scheme, parsed.netloc, parsed.path.strip("/")]
			return "_".join(piece.replace("/", "_") for piece in pieces if piece) or "stream"
		path = Path(source)
		if path.name == "img1" and path.parent.name:
			return path.parent.name
		if path.suffix:
			return path.stem
		return path.name or "run"

	def _resolve_output_fps(self) -> int:
		fps = getattr(self.args, "fps", None)
		if fps is not None:
			return int(fps)

		source = getattr(self.args, "source", None)
		if isinstance(source, (str, Path)):
			source_str = str(source)
			if not source_str.isdigit() and "://" not in source_str:
				path = Path(source_str)
				if path.is_file():
					capture = cv2.VideoCapture(str(path))
					try:
						video_fps = capture.get(cv2.CAP_PROP_FPS)
					finally:
						capture.release()
					if video_fps and video_fps > 0:
						return int(video_fps)

		return 30

	@staticmethod
	def initialize_trackers(predictor, args):
		tracker_name = str(getattr(args, "tracker", "")).lower()
		if tracker_name not in TRACKER_MAPPING:
			available = ", ".join(sorted(TRACKER_MAPPING))
			raise ValueError(f"'{tracker_name}' is not supported. Supported ones are {available}")

		reid_weights = _primary_model_ref(getattr(args, "reid", None))
		if reid_weights is not None:
			reid_weights = Path(reid_weights)

		batch_size = int(getattr(getattr(predictor, "dataset", None), "bs", 1) or 1)
		predictor.trackers = [
			TrackerRuntime.create(
				tracking_method=tracker_name,
				reid_weights=reid_weights,
				device=select_device(getattr(predictor, "device", "cpu")),
				half=bool(getattr(args, "half", False)),
				per_class=bool(getattr(args, "per_class", False)),
				target_id=getattr(args, "target_id", None),
			)
			for _ in range(batch_size)
		]
		return predictor.trackers

	def run(self):
		model = Boxmot(
			detector=_primary_model_ref(getattr(self.args, "detector", None)),
			reid=_primary_model_ref(getattr(self.args, "reid", None)),
			tracker=getattr(self.args, "tracker", get_mode_default("track", "tracker")),
			classes=getattr(self.args, "classes", None),
			project=getattr(self.args, "project", get_mode_default("track", "project")),
		)
		result = model.track(
			source=getattr(self.args, "source", get_mode_default("track", "source")),
			imgsz=getattr(self.args, "imgsz", None),
			conf=getattr(self.args, "conf", None),
			iou=float(getattr(self.args, "iou", get_mode_default("track", "iou"))),
			device=getattr(self.args, "device", get_mode_default("track", "device")),
			half=bool(getattr(self.args, "half", get_mode_default("track", "half"))),
			save=bool(getattr(self.args, "save", False)),
			save_txt=bool(getattr(self.args, "save_txt", False)),
			verbose=bool(getattr(self.args, "verbose", False)),
		)
		if getattr(self.args, "show", False):
			result.show()
		elif self._should_consume_result():
			previous_cache_results = getattr(result.results, "_cache_results", True)
			try:
				result.results._cache_results = False
				for _ in result.results:
					pass
			finally:
				result.results._cache_results = previous_cache_results
			result.refresh()
		return result


def main(args):
	return TrackingSession(args).run()


__all__ = ("TrackerRuntime", "TrackingSession", "TimingStats", "main", "wrap_tracker_reid")
