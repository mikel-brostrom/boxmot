# DeepStream Integration

BoxMOT provides a native DeepStream adapter (`libnvds_boxmot_tracker.so`) that allows you to use BoxMOT's tracking algorithms (BoTSORT, ByteTrack, OCSORT, SFSORT, OccluBoost) inside NVIDIA DeepStream pipelines as a drop-in replacement for DeepStream's built-in trackers (IOU, NvSORT, NvDCF, NvDeepSORT).

## Features

- **Full NvDsTracker API** — implements `NvMOT_Query`, `NvMOT_Init`, `NvMOT_Process`, `NvMOT_DeInit`, `NvMOT_RemoveStreams`, `NvMOT_RetrieveMiscData`
- **TensorRT-accelerated ReID** — matches DeepStream's native ReID inference pipeline exactly (same preprocessing, same model format support)
- **Multi-stream batch processing** — per-stream tracker instances with shared batched GPU inference
- **DeepStream-compatible config** — YAML format matching NvMultiObjectTracker conventions
- **All BoxMOT algorithms** — choose between BoTSORT, ByteTrack, OCSORT, SFSORT, or OccluBoost

## Requirements

| Component | Version |
|-----------|---------|
| NVIDIA DeepStream SDK | 6.0+ |
| TensorRT | 8.0+ |
| CUDA Toolkit | 11.0+ |
| OpenCV | 4.0+ |
| Eigen3 | 3.3+ |
| CMake | 3.16+ |

## Building

### From Python (recommended)

```python
from boxmot.native.deepstream_adapter import build_adapter

# Auto-detects DeepStream and TensorRT installations
lib_path = build_adapter()
print(f"Built: {lib_path}")
```

### From command line

```bash
cd boxmot/native/trackers/deepstream
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEEPSTREAM_ROOT=/opt/nvidia/deepstream/deepstream \
    -DTENSORRT_ROOT=/usr/local/tensorrt
make -j$(nproc)
```

The output is `libnvds_boxmot_tracker.so`.

## Usage in DeepStream

### App config (`deepstream_app_config.txt`)

```ini
[tracker]
enable=1
tracker-width=960
tracker-height=544
ll-lib-file=/path/to/libnvds_boxmot_tracker.so
ll-config-file=/path/to/config_tracker_boxmot.yml
gpu-id=0
display-tracking-id=1
```

### Tracker config (`config_tracker_boxmot.yml`)

```yaml
BaseConfig:
    algorithm: botsort
    frameRate: 30

BoxMOT:
    trackHighThresh: 0.6
    trackLowThresh: 0.1
    newTrackThresh: 0.7
    trackBuffer: 30
    matchThresh: 0.8
    withReId: true

TargetManagement:
    maxTargetsPerStream: 150
    maxShadowTrackingAge: 30

ReID:
    reidType: 1
    onnxFile: "/path/to/reid_model.onnx"
    modelEngineFile: "/path/to/reid_model.engine"
    batchSize: 100
    networkMode: 1   # FP16
    inferDims: [256, 128, 3]
    netScaleFactor: 1.0
    offsets: [0.0, 0.0, 0.0]
    reidFeatureSize: 256
    addFeatureNormalization: 1
```

## Generating Config from Python

```python
from boxmot.native.deepstream_adapter import generate_config

config_path = generate_config(
    algorithm="botsort",
    reid_onnx="/opt/nvidia/deepstream/deepstream/samples/models/Tracker/resnet50_market1501_aicity156.onnx",
    reid_feature_size=256,
    reid_network_mode=1,  # FP16
    track_high_thresh=0.6,
    output_path="/path/to/config_tracker_boxmot.yml"
)
```

## ReID Model Compatibility

The TensorRT ReID backend uses the **same preprocessing pipeline** as DeepStream's NvMultiObjectTracker:

```
y = netScaleFactor * (x - offsets)
```

This means any ReID model trained for or used with DeepStream's NvDeepSORT/NvDCF trackers works directly with BoxMOT's DeepStream adapter. Supported model formats:

| Format | Config Key | Notes |
|--------|-----------|-------|
| ONNX | `onnxFile` | Recommended. Works with OSNet, LMBN, custom models |
| TAO Toolkit (ETLT) | `tltEncodedModel` + `tltModelKey` | NVIDIA pre-trained models |
| Pre-built TRT Engine | `modelEngineFile` | Fastest startup (skip engine build) |

### Example models

| Model | Feature Size | Input Dims | netScaleFactor | Notes |
|-------|-------------|-----------|----------------|-------|
| ReIdentificationNet (ResNet-50) | 256 | [256, 128, 3] | 1.0 | NVIDIA NGC, market1501 |
| OSNet x0.25 | 512 | [256, 128, 3] | 0.00392 (1/255) | MSMT17 |
| LMBN-n | 2048 | [384, 128, 3] | 0.00392 (1/255) | DukeMTMC |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepStream Pipeline                        │
│                                                              │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │  PGIE   │───▶│ Gst-nvtracker │───▶│  Downstream     │    │
│  │(Detector)│    │   (Plugin)    │    │  (OSD/Sink/etc) │    │
│  └─────────┘    └──────┬───────┘    └─────────────────┘    │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
              NvDsTracker API calls
                          │
              ┌───────────▼──────────────┐
              │  libnvds_boxmot_tracker  │
              │                          │
              │  ┌────────────────────┐  │
              │  │ TensorRT ReID Model│  │  ◀── Batched GPU inference
              │  │ (FP16/INT8)       │  │      across all streams
              │  └────────────────────┘  │
              │                          │
              │  ┌────────────────────┐  │
              │  │ Per-Stream Trackers│  │  ◀── BoTSORT / ByteTrack /
              │  │ Stream 0: BoTSORT │  │      OCSORT / SFSORT /
              │  │ Stream 1: BoTSORT │  │      OccluBoost
              │  │ Stream N: BoTSORT │  │
              │  └────────────────────┘  │
              │                          │
              └──────────────────────────┘
```

## Performance Characteristics

| Feature | BoxMOT Adapter | NvDCF | NvDeepSORT | NvSORT |
|---------|---------------|-------|-----------|--------|
| Visual tracking (DCF) | ✗ | ✓ | ✗ | ✗ |
| ReID features | ✓ (TensorRT) | ✓ (TensorRT) | ✓ (TensorRT) | ✗ |
| Kalman filter | ✓ | ✓ | ✓ | ✓ |
| CMC (camera motion) | ✓ (BoTSORT) | ✗ | ✗ | ✗ |
| Multi-stage assoc. | ✓ (ByteTrack) | ✗ | ✓ | ✗ |
| OBB support | ✓ | ✗ | ✗ | ✗ |
| Tracker confidence | ✗ | ✓ | ✗ | ✗ |
| GPU compute for tracking | ✗ (CPU) | ✓ | ✗ | ✗ |
| Past-frame output | ✓ | ✓ | ✓ | ✓ |

## Limitations

- **No GPU-accelerated visual tracking** — BoxMOT trackers run association logic on CPU. The GPU is used only for TensorRT ReID inference. For DCF-based visual tracking, use NvDCF.
- **No tracker confidence** — BoxMOT trackers don't generate per-object tracker confidence scores (always 1.0).
- **Linux + NVIDIA GPU only** — DeepStream requires Jetson or dGPU platforms.
- **No VPI integration** — The adapter doesn't use NVIDIA VPI for crop scaling; it uses OpenCV on CPU.
