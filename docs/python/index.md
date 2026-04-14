# Python API

Use `boxmot` for the high-level workflow facade, and explicit modules such as `boxmot.api`, `boxmot.reid`, `boxmot.trackers`, and `boxmot.trackers.tracker_zoo` for lower-level access.

For a getting-started view, see [Python Usage](../usage/python.md).

## Main entry points

- `Boxmot` for high-level workflow orchestration across `track`, `generate`, `val`, `tune`, `research`, and `export`
- `track(...)` for composable detector + ReID + tracker execution
- `evaluate(...)` for runtime summaries
- `Detector` for public detector wrapping
- `ReID` for the unified ReID runtime
- result types such as `TrackRunResult`, `GenerateResult`, `ValidationResult`, `TuneResult`, and `ResearchResult`

## Minimal example

```python
from boxmot import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run)
```

## Pages

- [Boxmot Facade](boxmot.md)
- [track(...) and evaluate(...)](functions.md)
- [Detector](detector.md)
- [ReID](reid.md)
- [Results Objects](results.md)
- [Tracker Factory](tracker-zoo.md)
