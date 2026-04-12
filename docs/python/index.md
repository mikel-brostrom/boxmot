# Python API

The public Python API is re-exported from `boxmot/__init__.py`.

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
from boxmot import Boxmot

model = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = model.track(source="video.mp4", save=True)
print(run.summary)
```

## Pages

- [Boxmot Facade](boxmot.md)
- [track(...) and evaluate(...)](functions.md)
- [Detector](detector.md)
- [ReID](reid.md)
- [Results Objects](results.md)
- [Tracker Factory](tracker-zoo.md)
