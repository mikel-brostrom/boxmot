# Boxmot Facade

Use `Boxmot` when you want the Python equivalent of the CLI with minimal boilerplate.

```python
from boxmot.api import Boxmot

boxmot = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = boxmot.track(source="video.mp4", save=True)
metrics = boxmot.val(benchmark="mot17-mini")
tuned = boxmot.tune(benchmark="mot17-mini", n_trials=2)

print(run.summary)
print(metrics.summary)
print(tuned.summary)
```

::: boxmot.api.Boxmot
