# Python

Use `boxmot` for the high-level workflow facade, and explicit modules such as `boxmot.reid`, `boxmot.trackers`, and `boxmot.trackers.tracker_zoo` when you want lower-level control.

## High-level facade

Use `Boxmot` when you want the Python equivalent of the CLI with minimal boilerplate:

```python
from boxmot import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
print(run)

cache = Boxmot().generate(benchmark="mot17-mini")
print(cache.cache_dir)

metrics = boxmot.val(benchmark="mot17-mini")
print(metrics)

tuned = boxmot.tune(benchmark="mot17-mini", n_trials=2)
print(tuned)
```

The same facade also exposes `research(...)` for GEPA-backed benchmark optimization and `export(...)` for ReID conversion workflows.

Use `.summary`, `.timings`, `.delta_summary`, or `.to_dict()` when you need structured data instead of the human-readable report.

`generate(...)` and `research(...)` return `GenerateResult` and `ResearchResult`.

## Composable runtime

If you need more control, compose the detector, ReID runtime, and tracker explicitly:

```python
from boxmot import track
from boxmot.reid import ReID
from boxmot.trackers import StrongSort
from boxmot.detectors import Detector

detector = Detector("yolov8n.pt", device="cpu")
reid = ReID("osnet_x0_25_msmt17.pt", device="cpu")
tracker = StrongSort(reid_weights="osnet_x0_25_msmt17.pt", device="cpu", half=False)

results = track("video.mp4", detector, reid, tracker, verbose=False)
print(results.summary())
```

## Result objects

High-level workflows return structured result types such as `TrackRunResult`, `GenerateResult`, `ValidationResult`, `TuneResult`, and `ResearchResult` rather than raw terminal text.

## Detailed API pages

- [Python API Overview](../python/index.md)
- [Boxmot Facade](../python/boxmot.md)
- [track(...) and evaluate(...)](../python/functions.md)
- [Results Objects](../python/results.md)
