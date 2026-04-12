# Python

The public Python API is re-exported from `boxmot`, so most user code should import directly from the package root.

## High-level facade

Use `Boxmot` when you want the Python equivalent of the CLI with minimal boilerplate:

```python
from boxmot import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run.summary)
```

The same facade also exposes `val(...)`, `tune(...)`, and `export(...)`.

## Composable runtime

If you need more control, compose the detector, ReID runtime, and tracker explicitly:

```python
from boxmot import ReID, StrongSort, track
from boxmot.detectors import Detector

detector = Detector("yolov8n.pt", device="cpu")
reid = ReID("osnet_x0_25_msmt17.pt", device="cpu")
tracker = StrongSort(reid_weights="osnet_x0_25_msmt17.pt", device="cpu", half=False)

results = track("video.mp4", detector, reid, tracker, verbose=False)
print(results.summary())
```

## Result objects

Tracking and evaluation calls return structured result types such as `TrackRunResult`, `ValidationResult`, and `TuneResult` rather than raw terminal text.

## Detailed API pages

- [Python API Overview](../python/index.md)
- [Boxmot Facade](../python/boxmot.md)
- [track(...) and evaluate(...)](../python/functions.md)
- [Results Objects](../python/results.md)
