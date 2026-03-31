import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import cv2
import numpy as np

from boxmot.utils.mot_utils import (
    convert_to_mmot_obb_format,
    convert_to_mot_format,
    format_mot_results,
    xywha_to_corners,
)

try:
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    Annotator = None


def _ensure_2d_tracks(tracks: np.ndarray) -> np.ndarray:
    """Normalize tracking output to a 2D array."""
    arr = np.asarray(tracks)
    if arr.size == 0:
        if arr.ndim == 2:
            return arr
        return np.empty((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _serialize_tracks(tracks: np.ndarray, frame_id: int) -> str:
    """Serialize one frame of tracker output to MOT-style text."""
    tracks = _ensure_2d_tracks(tracks)
    if tracks.size == 0:
        return ""
    if tracks.shape[1] >= 9:
        mot_rows = convert_to_mmot_obb_format(tracks, frame_id)
    else:
        mot_rows = convert_to_mot_format(tracks, frame_id)
    return format_mot_results(mot_rows)


class Tracks:
    """Structured per-frame tracking result."""

    def __init__(
        self,
        frame_id: int,
        frame: np.ndarray,
        tracks: np.ndarray,
        get_drawer: Callable[[], Callable],
    ) -> None:
        self.frame_id = frame_id
        self.frame = frame
        self.tracks = _ensure_2d_tracks(tracks)
        self._get_drawer = get_drawer

    @property
    def is_obb(self) -> bool:
        return bool(self.tracks.ndim == 2 and self.tracks.shape[1] >= 9)

    @property
    def xyxy(self) -> np.ndarray:
        """Return AABB geometry or an enclosing AABB for OBB results."""
        if self.tracks.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if not self.is_obb:
            return self.tracks[:, :4]

        corners = xywha_to_corners(self.tracks[:, :5]).reshape(-1, 4, 2)
        mins = corners.min(axis=1)
        maxs = corners.max(axis=1)
        return np.concatenate((mins, maxs), axis=1)

    @property
    def xywha(self) -> Optional[np.ndarray]:
        """Return OBB geometry when available."""
        if not self.is_obb:
            return None
        return self.tracks[:, :5]

    @property
    def id(self) -> np.ndarray:
        if self.tracks.size == 0:
            return np.empty((0,), dtype=np.int32)
        return self.tracks[:, 5 if self.is_obb else 4].astype(np.int32)

    @property
    def ids(self) -> np.ndarray:
        return self.id

    @property
    def conf(self) -> np.ndarray:
        if self.tracks.size == 0:
            return np.empty((0,), dtype=np.float32)
        return self.tracks[:, 6 if self.is_obb else 5]

    @property
    def cls(self) -> np.ndarray:
        if self.tracks.size == 0:
            return np.empty((0,), dtype=np.int32)
        return self.tracks[:, 7 if self.is_obb else 6].astype(np.int32)

    @property
    def det_ind(self) -> np.ndarray:
        if self.tracks.size == 0:
            return np.empty((0,), dtype=np.int32)
        index = 8 if self.is_obb else 7
        if self.tracks.shape[1] <= index:
            return -np.ones((len(self.tracks),), dtype=np.int32)
        return self.tracks[:, index].astype(np.int32)

    def to_mot(self) -> str:
        """Serialize this frame to MOT-style text."""
        return _serialize_tracks(self.tracks, self.frame_id)

    def __len__(self) -> int:
        return len(self.tracks)

    def __repr__(self) -> str:
        geometry = "xywha" if self.is_obb else "xyxy"
        return f"Tracks(frame_id={self.frame_id}, tracks={len(self)}, geometry={geometry!r})"
        
    def show(self) -> bool:
        drawer = self._get_drawer()
        
        if drawer is None:
             drawn_frame = self._default_draw(self.frame, self.tracks)
        else:
             drawn_frame = drawer(self.frame, self.tracks)
             
        cv2.imshow("Tracking", drawn_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
             return False
        return True

    def _default_draw(self, frame, tracks):
        if Annotator is None:
            return frame
        
        annotator = Annotator(frame, line_width=2, example=str(" "))
        for bbox, track_id, conf in zip(self.xyxy, self.id, self.conf):
            # cls maps to coco classes, but let's check length
            # Some trackers might output differently. 
            # Standard BoxMOT output is: [x1, y1, x2, y2, id, conf, cls, det_ind]
            
            label = f"{track_id} {conf:.2f}"
            annotator.box_label(bbox, label, color=colors(int(track_id), True))
        return annotator.result()


class Results:
    def __init__(self, source: Union[str, int, Path], detector: Any, reid: Any, tracker: Any, verbose: bool = True):
        self.source = source
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.verbose = verbose
        self.drawer = None 
        self._generator = None
        self._cache = []
        
        # Timing accumulators
        self.totals = {
            'det': 0.0,
            'reid': 0.0,
            'track': 0.0,
            'total': 0.0,
            'frames': 0
        }
        
    def __iter__(self) -> Iterator[Tracks]:
        if self._generator is None:
            self._generator = self._process()
        return self

    def __next__(self) -> Tracks:
        if self._generator is None:
             self._generator = self._process()
        tracks = next(self._generator)
        self._cache.append(tracks)
        return tracks
    
    def _log_frame_timings(self, frame_num: int, det_time: float, reid_time: float, track_time: float) -> None:
        """Log timing information for a single frame."""
        total = det_time + reid_time + track_time
        
        log_msg = f"Frame {frame_num} | Det: {det_time:.1f}ms"
        if self.reid:
            log_msg += f" | ReID: {reid_time:.1f}ms"
        log_msg += f" | Track: {track_time:.1f}ms | Total: {total:.1f}ms"
        
        print(log_msg)
        
        # Accumulate
        self.totals['det'] += det_time
        self.totals['reid'] += reid_time
        self.totals['track'] += track_time
        self.totals['total'] += total
        self.totals['frames'] += 1

    def _print_summary(self):
        """Print execution time summary table."""
        if self.totals['frames'] == 0:
            return

        frames = self.totals['frames']
        det_avg = self.totals['det'] / frames
        reid_avg = self.totals['reid'] / frames
        track_avg = self.totals['track'] / frames
        total_avg = self.totals['total'] / frames
        
        print("\n" + "="*65)
        print(f"{'Component':<15} | {'Total Time (ms)':<20} | {'Average Time (ms)':<20}")
        print("-" * 65)
        print(f"{'Detection':<15} | {self.totals['det']:<20.1f} | {det_avg:<20.1f}")
        if self.reid:
            print(f"{'ReID':<15} | {self.totals['reid']:<20.1f} | {reid_avg:<20.1f}")
        print(f"{'Tracking':<15} | {self.totals['track']:<20.1f} | {track_avg:<20.1f}")
        print("-" * 65)
        print(f"{'Total':<15} | {self.totals['total']:<20.1f} | {total_avg:<20.1f}")
        print("="*65 + "\n")

    def _get_frames(self):
        src = self.source
        # Check source type for directory
        if isinstance(src, (str, Path)) and Path(src).is_dir():
            path_src = Path(src)
            exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
            images = []
            for ext in exts:
                images.extend(path_src.glob(ext))
            images = sorted(images)
            
            if not images:
                raise ValueError(f"No images found in {src}")
            
            for img_path in images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                yield frame
            return

        # Video source logic
        if isinstance(src, (str, Path)):
             if str(src).isdigit():
                  src = int(src)
             else:
                  src = str(src)
        
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
             raise ValueError(f"Could not open video source: {src}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def _process(self):
        frame_generator = self._get_frames()
        frame_num = 0
        
        try:
            for frame in frame_generator:
                frame_num += 1
                
                # 1. Detect
                t0 = time.time()
                dets = self.detector(frame)
                det_time = (time.time() - t0) * 1000
                
                # 2. ReID
                if self.reid:
                    t1 = time.time()
                    features = self.reid(frame, dets)
                    reid_time = (time.time() - t1) * 1000
                else:
                    features = None
                    reid_time = 0.0
                     
                # 3. Track
                t2 = time.time()
                tracks = self.tracker.update(dets, frame, features)
                track_time = (time.time() - t2) * 1000
                
                # Log timings (accumulates automatically)
                if self.verbose:
                    self._log_frame_timings(frame_num, det_time, reid_time, track_time)
                
                yield Tracks(frame_num, frame, tracks, get_drawer=lambda: self.drawer)
                
        finally:
            if self.verbose:
                self._print_summary()

    def show(self):
        for track_result in self:
            if not track_result.show():
                break
        cv2.destroyAllWindows()

    def save(self, output_path: Union[str, Path]) -> Path:
        """Save cached and remaining streamed tracking output to a text file."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w") as handle:
            for track_result in self._cache:
                handle.write(track_result.to_mot())
            for track_result in self:
                handle.write(track_result.to_mot())

        return destination


def track(source, detector, reid, tracker, verbose: bool = True) -> Results:
    return Results(source, detector, reid, tracker, verbose=verbose)
