import cv2
import numpy as np
from pathlib import Path
from typing import Union, Iterator, Callable, Any, Optional
import time

try:
    from ultralytics.utils.plotting import Annotator, colors
except ImportError:
    Annotator = None

class Tracks:
    def __init__(self, frame: np.ndarray, tracks: np.ndarray, get_drawer: Callable[[], Callable]):
        self.frame = frame
        self.tracks = tracks
        self._get_drawer = get_drawer
        
    def show(self):
        drawer = self._get_drawer()
        # If drawer is None or not callable, we might fallback or error?
        # User snippet implies drawer is a function taking (frame, tracks).
        # We will implement a default drawer if one isn't provided.
        
        if drawer is None:
             # Fallback default drawer using Annotator
             drawn_frame = self._default_draw(self.frame, self.tracks)
        else:
             drawn_frame = drawer(self.frame, self.tracks)
             
        cv2.imshow("Tracking", drawn_frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
             pass

    def _default_draw(self, frame, tracks):
        if Annotator is None:
            return frame
        
        annotator = Annotator(frame, line_width=2, example=str(" "))
        for t in tracks:
            # tracks: [x1, y1, x2, y2, id, conf, cls, det_ind]
            bbox = t[:4]
            id = int(t[4])
            conf = t[5]
            cls = int(t[6])
            
            label = f"{id} {conf:.2f}"
            annotator.box_label(bbox, label, color=colors(id, True))
        return annotator.result()


class Results:
    def __init__(self, source: Union[str, int, Path], detector: Any, reid: Any, tracker: Any, verbose: bool = True):
        self.source = source
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.verbose = verbose
        self.drawer = None # Can be set by user
        self._generator = None
        self._cache = []
        
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
        print(f"Frame {frame_num} | Det: {det_time:.1f}ms | ReID: {reid_time:.1f}ms | Track: {track_time:.1f}ms | Total: {total:.1f}ms")

    def _process(self):
        # Handle source
        # checks if source is int (webcam) or str (file)
        src = self.source
        if isinstance(src, (str, Path)):
             if str(src).isdigit():
                  src = int(src)
             else:
                  src = str(src)
        
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
             raise ValueError(f"Could not open video source: {src}")
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # 1. Detect
            t0 = time.time()
            dets = self.detector(frame)
            det_time = (time.time() - t0) * 1000
            
            # 2. ReID
            t1 = time.time()
            features = self.reid(frame, dets) if self.reid else None
            reid_time = (time.time() - t1) * 1000
                 
            # 3. Track
            t2 = time.time()
            tracks = self.tracker.update(dets, frame, features)
            track_time = (time.time() - t2) * 1000
            
            # Log timings
            if self.verbose:
                self._log_frame_timings(frame_num, det_time, reid_time, track_time)
            
            yield Tracks(frame, tracks, get_drawer=lambda: self.drawer)
            
        cap.release()

    def show(self):
        for track_result in self:
            track_result.show()
        cv2.destroyAllWindows()


def track(source, detector, reid, tracker, verbose: bool = True) -> Results:
    return Results(source, detector, reid, tracker, verbose=verbose)
