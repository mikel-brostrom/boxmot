# Detectors

This folder contains BoxMOT's detector runtime layer.

If you want to bring your own detector into BoxMOT, there are two different paths:

1. Your weights already belong to a supported detector family such as Ultralytics YOLO, YOLOX, or RT-DETR.
2. You want to add a completely new detector backend family.

The first path usually only needs a weights file and, optionally, a detector YAML config. The second path requires a new backend module and a registry entry.

## Folder layout

```text
boxmot/detectors/
  README.md         # this guide
  __init__.py       # public detector exports
  base.py           # Detections dataclass and BaseDetectorBackend contract
  detector.py       # public Detector wrapper used by workflows and API
  registry.py       # backend routing and detector-config lookup
  ultralytics.py    # Ultralytics backend
  yolox.py          # YOLOX backend
  rtdetr.py         # RT-DETR backend
```

## Related folders

```text
boxmot/configs/detectors/    # detector YAML configs
models/                      # conventional place to keep detector weights
tests/unit/                  # detector and inference tests
```

Important files around this folder:

- `boxmot/configs/detectors/README.md` explains detector config resolution.
- `boxmot/configs/detectors/<family>/*.yaml` stores detector-specific defaults such as `imgsz`, `conf`, `box_type`, and optional download URLs.
- `models/` is the repo's default location for detector weights, but BoxMOT can also use explicit paths outside that directory.

## How BoxMOT chooses a detector backend

The public `Detector` wrapper in `detector.py` calls `get_detector_class(path)` from `registry.py`.

`registry.py` decides which backend to instantiate by checking the detector filename against known family markers.

Current built-in families are:

- Ultralytics: filenames containing `yolov8`, `yolov9`, `yolov10`, `yolo11`, `yolo12`, `yolo26`, or `sam`
- YOLOX: filenames containing `yolox_n`, `yolox_s`, `yolox_m`, `yolox_l`, or `yolox_x`
- RT-DETR: filenames containing `rtdetr_v2_r50vd`, `rtdetr_v2_r18vd`, or `rtdetr_v2_r101vd`

That means a custom weights file for a supported family should keep one of those family markers in its filename, otherwise BoxMOT will not know which backend to load.

## Option 1: Bring your own weights for a supported family

If your detector already matches one of the supported families, you usually do not need to add new Python code.

### 1. Put the weights somewhere accessible

The conventional choice is:

```text
models/my_detector_weights.pt
```

Any explicit path works, but keeping weights under `models/` matches the rest of the repository.

### 2. Keep the family marker in the filename

Examples:

- `models/yolo11_custom_people.pt`
- `models/yolox_x_my_benchmark.pt`
- `models/rtdetr_v2_r50vd_traffic.pt`

This is what allows `registry.py` to select the correct backend.

### 3. Optionally add a detector config

Detector configs live under:

```text
boxmot/configs/detectors/<family>/
```

Example:

```yaml
id: yolo11_custom_people
model: models/yolo11_custom_people.pt
url: https://example.com/yolo11_custom_people.pt
imgsz: [800, 1440]
conf: 0.20
box_type: aabb
classes:
  0: person
```

Why this helps:

- `model` enables exact-match lookup for your weights file
- `url` allows automatic download when the file is missing
- `imgsz` and `conf` provide detector-specific defaults
- `box_type` tells BoxMOT whether the detector emits AABB or OBB detections

Config lookup behavior is:

1. exact match on `model` or `default_model`
2. fallback to a family default such as `ultralytics/default.yaml`

See `boxmot/configs/detectors/README.md` for the full detector-config details.

### 4. Run BoxMOT with your weights

Example:

```bash
boxmot track --source path/to/video.mp4 --detector models/yolo11_custom_people.pt
boxmot eval --benchmark mot17-ablation --detector models/yolo11_custom_people.pt
```

## Option 2: Add a brand-new detector backend family

If your detector does not fit one of the existing families, add a new backend module under this folder and register it in `registry.py`.

### 1. Create a backend module

Add a file such as:

```text
boxmot/detectors/mydetector.py
```

Implement a backend class that follows the `BaseDetectorBackend` contract from `base.py`.

### 2. Return `Detections` objects in the BoxMOT schema

BoxMOT expects each prediction result to be wrapped in the `Detections` dataclass.

Supported detection layouts are:

- AABB: `(N, 6)` with `[x1, y1, x2, y2, conf, cls]`
- OBB: `(N, 7)` with `[cx, cy, w, h, angle, conf, cls]`

Minimal backend skeleton:

```python
from __future__ import annotations

import numpy as np

from boxmot.detectors.base import BaseDetectorBackend, Detections


class MyDetector(BaseDetectorBackend):
    def __init__(self, model, device, imgsz=None):
        self.device = device
        self.imgsz = imgsz
        self.model = self._load_model(model)
        self.names = {0: "person"}

    def _load_model(self, model):
        return model

    def __call__(self, images: list, conf, iou, classes, agnostic_nms) -> list[Detections]:
        results = []
        for image in images:
            dets = np.empty((0, 6), dtype=np.float32)
            results.append(
                Detections(
                    dets=dets,
                    orig_img=image,
                    path="",
                    names=self.names,
                )
            )
        return results
```

In practice, your backend can either:

- implement `__call__` directly, like `UltralyticsDetector`
- or implement `preprocess`, `process`, and `postprocess`, like `YoloXDetector`

### 3. Register the backend in `registry.py`

Add a detector-family check and a new registry entry in `get_detector_class()`.

At minimum you need:

- a matcher that identifies your model filenames
- optional package requirements to auto-install
- the module path
- the class name

If you want detector-family defaults, also extend `_DEFAULT_DETECTOR_MAP` and `_model_family()`.

### 4. Add detector configs if you want config-driven defaults

If your backend should support per-model defaults or downloads, add YAML files under:

```text
boxmot/configs/detectors/mydetector/
```

This is optional, but recommended if users should be able to run the backend with only a model name and sane defaults.

### 5. Keep the public wrapper unchanged

The `Detector` class in `detector.py` already handles:

- source iteration
- batching
- warmup
- callbacks
- returning raw arrays or `Detections`

A new backend usually does not need to reimplement that workflow layer. Focus on model loading and batched inference.

## Recommended smoke checks

After adding a new detector or config:

```bash
uv run python -m boxmot.engine.cli track --source path/to/image_or_video --detector path/to/weights.pt
uv run python -m boxmot.engine.cli eval --benchmark mot17-ablation --detector path/to/weights.pt
uv run pytest tests/unit/test_base_backend.py tests/unit/test_inference.py
```

## Practical summary

- New weights for an existing family: usually add weights, optionally add a detector YAML, keep the family marker in the filename.
- New detector family: add a backend module in this folder and register it in `registry.py`.
- Detector defaults live under `boxmot/configs/detectors/`.
- Weights conventionally live under `models/`.