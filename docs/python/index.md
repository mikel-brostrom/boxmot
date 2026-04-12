# Python API

Use the explicit public modules: `boxmot.api`, `boxmot.reid`, `boxmot.trackers`, and `boxmot.trackers.tracker_zoo`. The package root is intentionally limited to metadata.

For a getting-started view, see [Python Usage](../usage/python.md).

## Main entry points

- `Boxmot` for high-level workflow orchestration
- `track(...)` for composable detector + ReID + tracker execution
- `evaluate(...)` for runtime summaries
- `Detector` for public detector wrapping
- `ReID` for the unified ReID runtime
- result types such as `TrackRunResult`, `ValidationResult`, and `TuneResult`

## Minimal example

```python
from boxmot.api import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run.summary)
```

## Pages

- [Boxmot Facade](boxmot.md)
- [track(...) and evaluate(...)](functions.md)
- [Detector](detector.md)
- [ReID](reid.md)
- [Results Objects](results.md)
- [Tracker Factory](tracker-zoo.md)
