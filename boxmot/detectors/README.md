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
  config.py         # detector profile validation and runtime adaptation
  detector.py       # public Detector wrapper used by workflows and API
  registry.py       # backend routing and detector-config lookup
  ultralytics.py    # Ultralytics backend
  yolox.py          # YOLOX backend
  rtdetr.py         # RT-DETR backend
```

## Related folders

```text
boxmot/configs/detectors/       # central detector runtime profiles
models/                         # conventional place to keep detector weights
tests/unit/detectors/           # detector backend tests
tests/unit/engine/tracking/     # inference workflow tests
```

Important files around this folder:

- `docs/config/detectors.md` explains the central detector profile contract.
- `boxmot/configs/detectors/<profile>.yaml` stores box type, classes,
  inference defaults, and named checkpoints for a reusable detector profile.
- `models/` is the repo's default location for detector weights, but BoxMOT can also use explicit paths outside that directory.

## How BoxMOT chooses a detector backend

The public `Detector` wrapper in `detector.py` calls `get_detector_class(path)` from `registry.py`.

`registry.py` decides which backend to instantiate by checking the detector filename against known family markers.
Matching is case-insensitive and only examines the filename, not its parent directories.

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
boxmot/configs/detectors/
```

Example:

```yaml
id: yolo11-custom-people
box_type: aabb
classes:
  0: person

inference:
  image_size: [800, 1440]
  confidence_threshold: 0.20

checkpoints:
  default:
    path: models/yolo11_custom_people.pt
    uri: https://example.com/yolo11_custom_people.pt
```

Why this helps:

- a named checkpoint enables exact-match lookup for your weights file
- `uri` allows automatic download when the file is missing
- `image_size` and `confidence_threshold` provide detector-specific defaults
- `box_type` tells BoxMOT whether the detector emits AABB or OBB detections

Direct detector lookup matches the requested model stem against checkpoint
paths in the catalog. Experiments reference the profile ID and checkpoint name
explicitly. See `docs/config/detectors.md` for the complete contract.

### 4. Run BoxMOT with your weights

Example:

```bash
boxmot track --source path/to/video.mp4 --detector models/yolo11_custom_people.pt
boxmot eval --experiment mot17-ablation-yolox-lmbn --detector models/yolo11_custom_people.pt
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

BoxMOT expects exactly one `Detections` result per input image. The dataclass validates
the detection width, normalizes values to `float32`, and verifies that masks stay
row-aligned with detections.

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
        self._images = []

    def _load_model(self, model):
        return model

    def preprocess(self, images: list[np.ndarray]):
        self._images = images
        return images

    def process(self, preprocessed):
        return self.model(preprocessed)

    def postprocess(self, predictions, conf, iou, classes, agnostic_nms):
        return [
            Detections.empty(image, names=self.names)
            for image in self._images
        ]
```

`BaseDetectorBackend.__call__` composes these three stages and checks the output
type and batch cardinality. Keep model decoding, confidence filtering, class
filtering, and NMS in `postprocess` so timing remains comparable across backends.

### 3. Register the backend in `registry.py`

Add a detector-family matcher and a `DetectorBackendSpec` entry to
`DETECTOR_BACKENDS`.

At minimum you need:

- a matcher that identifies your model filenames
- optional package requirements to auto-install
- the module path
- the class name

### 4. Add detector configs if you want config-driven defaults

If your backend should support per-model defaults or downloads, add YAML files under:

```text
boxmot/configs/detectors/
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
uv run python -m boxmot.engine.cli eval --experiment mot17-ablation-yolox-lmbn --detector path/to/weights.pt
uv run pytest tests/unit/detectors tests/unit/engine/tracking/test_inference.py
```

## Practical summary

- New weights for an existing family: usually add weights, optionally add a detector YAML, keep the family marker in the filename.
- New detector family: add a backend module in this folder and register it in `registry.py`.
- Detector profiles live under `boxmot/configs/detectors/`.
- Weights conventionally live under `models/`.
