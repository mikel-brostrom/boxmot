This directory contains optional runtime detector config files.

When a benchmark config is not supplying detector settings, BoxMOT looks for a
YAML file here whose stem matches the selected detector model name.

If both a benchmark detector config and a model-matched detector YAML are
present, the model YAML overrides the runtime fields for that detector.

Example:

- model: `models/yolo11s-obb.pt`
- config: `boxmot/configs/detectors/yolo11s-obb.yaml`

Supported fields:

- `imgsz`
- `conf`
- `classes`
