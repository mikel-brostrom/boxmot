# BoxMOT DeepStream Adapter

This directory implements a DeepStream-compatible low-level tracker library (`libnvds_boxmot_tracker.so`) that wraps BoxMOT's native C++ trackers (BoTSORT, ByteTrack, OCSORT, SFSORT, OccluBoost) behind NVIDIA's `NvDsTracker` API.

## Features

- Full implementation of `NvMOT_Query`, `NvMOT_Init`, `NvMOT_Process`, `NvMOT_DeInit`, `NvMOT_RemoveStreams`, `NvMOT_RetrieveMiscData`
- Multi-stream batch processing with per-stream tracker instances
- TensorRT-accelerated ReID inference matching DeepStream's native ReID pipeline
- Supports all BoxMOT tracker algorithms
- YAML configuration compatible with DeepStream's tracker config style

## Requirements

- NVIDIA DeepStream SDK (6.0+)
- TensorRT (8.0+)
- CUDA toolkit
- OpenCV 4+
- Eigen3

## Build

```bash
cd boxmot/native/trackers/deepstream
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage in DeepStream

In your DeepStream app config:

```ini
[tracker]
enable=1
tracker-width=960
tracker-height=544
ll-lib-file=/path/to/libnvds_boxmot_tracker.so
ll-config-file=config_tracker_boxmot.yml
```

## Configuration

See `config_tracker_boxmot.yml` for all available parameters.
