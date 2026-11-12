# Installation

BoxMOT supports Python **3.10–3.13**. Install it in a virtual environment, then
add the dependencies needed for your detector or workflow.

## Quick install

Install BoxMOT and the Ultralytics backend used in the tracking examples:

```bash
pip install boxmot "ultralytics>=8.4.58"
boxmot --help
boxmot track --source video.mp4 --detector yolov8n --tracker bytetrack --save
```

Replace `video.mp4` with your video. The first run downloads the detector
weights. See [Track](../modes/track.md) for webcams, streams, and output options.

If you supply your own detections, `pip install boxmot` provides the CLI,
Python API, trackers, and core ReID stack without a detector package. ReID
training, evaluation, and comparison also use this core installation.

The commands above use the standard PyPI PyTorch build. For CPU-only or CUDA
13.0 wheels, follow [PyTorch and source setup](pytorch.md).

## Mode-specific extras

Add optional dependencies to an installed BoxMOT environment with `install`:

```bash
boxmot install --extra onnx
boxmot install --extra evolve --extra research
```

| Workflow | Extra | Guide |
| --- | --- | --- |
| Ultralytics and YOLOX detectors (source checkout) | `yolo` | [Source installation](pytorch.md#source-checkout) |
| RT-DETR v2 detectors | `rtdetr` | [Detector models](../config/detectors.md) |
| Hyperparameter tuning | `evolve` | [Tune](../modes/tune.md) |
| Code-evolution research | `research` | [Research](../modes/research.md) |
| HTTP tracker service | `service` | [Service deployment](../guides/deployment.md) |
| Ollama or vLLM track descriptions | `attributes` | [Track Attributes](../config/attributes.md) |
| MOTS mask evaluation | `mots` | [EagerMOT](../trackers/eagermot.md) |
| Reference TrackEval comparison | `trackeval` | [Evaluation](../guides/evaluation.md) |
| ONNX export and inference | `onnx` | [Export](../modes/export.md) |
| Core ML export and inference | `coreml` | [Export](../modes/export.md) |
| OpenVINO export and inference | `onnx` and `openvino` | [Export](../modes/export.md) |
| TFLite export and inference | `tflite` | [Export](../modes/export.md) |

The `yolo` extra includes YOLOX, which uses the repository's dependency and
build overrides; use the source workflow when you need that backend. The quick
install above is sufficient for Ultralytics detectors such as `yolov8n`.

Core ML requires macOS. TensorRT needs a compatible CUDA/NVIDIA stack and
[separate installation steps](../modes/install.md#tensorrt). KITTI 2D evaluation
uses built-in metrics and does not require `trackeval`.

Dependency checks report an installation command when a workflow needs more
packages. To target a specific Python interpreter, run:

```bash
python -m boxmot.engine.cli install --extra onnx
```

See [Install dependencies](../modes/install.md) for custom requirements. For
source checkouts, continue using `uv sync` with the selected
[PyTorch profile](pytorch.md); `boxmot install` does not accept `cpu` or `cu130`.

## Other installation options

| Setup | Guide |
| --- | --- |
| Source checkout, CPU-only Torch, or CUDA 13.0 Torch | [PyTorch and source setup](pytorch.md) |
| Containerized CLI | [Docker](../integrations/docker.md) |
| Detection-to-track HTTP service | [Service deployment](../guides/deployment.md) |
| Native C++ tracker backends | [Native dependencies and builds](../native/index.md#building) |
| Development and documentation tools | [Contributing](../contributing/index.md) |

## Verify the Python API

```python
import boxmot
from boxmot.trackers import TrackerSpec

tracker = boxmot.create_tracker(TrackerSpec(name="bytetrack"))
print(boxmot.__version__, tracker.requirements)
```

Continue with the [overview](../index.md), [CLI workflows](../modes/index.md),
or [Python API](../python/index.md).
