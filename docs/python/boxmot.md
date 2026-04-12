# Boxmot Facade

Use `Boxmot` when you want the Python equivalent of the CLI with minimal boilerplate.

```python
from boxmot import Boxmot

model = Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
run = model.track(source="video.mp4", save=True)
metrics = model.val(benchmark="mot17-mini")
```

::: boxmot.api.Boxmot
