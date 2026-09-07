# Native Tracker Layout

BoxMOT keeps tracker-domain code separate from native implementation details:

- Python box-tracker implementations: representation-first packages at
  `boxmot/trackers/box/<name>/tracker.py`, with capabilities declared by each
  tracker
- Native C++ tracker family: `boxmot/native/cpp/trackers/<name>/`
- Native C++ model-free tracker base: `boxmot/native/cpp/trackers/base/`
- Independent native ReID runtime and C ABI: `boxmot/native/cpp/reid/`
- ABI-neutral native runtime helpers: `boxmot/native/cpp/include/boxmot/native/`
- Low-level ctypes binding: `boxmot/native/trackers/<name>.py`
- Canonical tracker adapter: `boxmot/trackers/box/<name>/native.py`
- Backend selection and validation: `boxmot/trackers/factory.py`

This keeps `boxmot.native` limited to C++ sources, build support, and raw ABI
bindings. Canonical Torch structures and tracker requirements remain in the
tracker domain.

## Organization

- `boxmot/native/cpp/trackers/base/`: shared association, assignment, and typed tracker-ABI primitives.
- `boxmot/native/cpp/reid/`: standalone appearance inference implementation; tracker targets must not include or link it.
- `boxmot/native/cpp/include/boxmot/native/`: low-level helpers shared across independent native libraries.
- `boxmot/native/cpp/trackers/<name>/include/<name>/`: public tracker-specific headers.
- `boxmot/native/cpp/trackers/<name>/src/`: tracker-specific implementation and ABI entrypoints.

Tracker-specific code should stay in its own directory even when two trackers
look similar. Only tracker primitives shared across algorithms belong in
`trackers/base`; general native runtime helpers belong under `cpp/include`, and
model inference belongs under its owning component subtree.

## Selection

Select the native backend in an immutable tracker specification:

```python
from boxmot import create_tracker
from boxmot.trackers import TrackerSpec

tracker = create_tracker(TrackerSpec(name="bytetrack", backend="cpp", geometry="aabb"))
```

## Build entrypoints

- Per-tracker builds still work from `boxmot/native/cpp/trackers/<name>/` and are what the low-level bindings load.
- The tree can also be configured from `boxmot/native/cpp/` to build the shared base and all registered native trackers together.

`boxmot_tracker_base` contains only association and assignment code. Tracker
targets do not link native ReID or ONNX Runtime, and do not request or directly
link OpenCV DNN. The aggregate build also exposes a separate
`boxmot_native_reid`/`reid_capi` target by default; pass
`-DBOXMOT_BUILD_NATIVE_REID=OFF` for a tracker-only aggregate build. Standalone
per-tracker builds default to tracker-only operation. Individual OpenCV
distributions may add their own transitive dependencies to modules required for
camera-motion compensation.

Build the native appearance encoder independently with:

```bash
cmake -S boxmot/native/cpp/reid -B build/native/reid -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/reid --target reid_capi
```

## Current scope

Typed v2 native backends are registered for:

- `botsort`
- `bytetrack`
- `occluboost`
- `ocsort`
- `sfsort`

Each tracker exposes an in-process shared library. Python streams keyed
Parquet rows through this live API; positional NPY replay executables are not
built or published.

Build requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

## ABI v2

The flat C ABI accepts separate float geometry, score, and embedding buffers
and separate `int64` class and detection-index buffers. Updates return a
library-allocated `BoxMOTTrackBatchV2`; callers must release it with the
tracker's `boxmot_<name>_result_free_v2` function. C++ tracker libraries never
load a ReID model: BotSort and OccluBoost consume embedding buffers supplied by
the high-level adapter or another caller. The Python-facing native adapters can
derive a missing buffer from a `Frame` before crossing this ABI.

## Embedding in a C++ Program

For a standalone C++ application, link against the tracker core target directly. Example for ByteTrack:

```cmake
add_subdirectory("${BOXMOT_ROOT}/boxmot/native/cpp/trackers/bytetrack" "${CMAKE_BINARY_DIR}/boxmot_bytetrack")
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE bytetrack_core)
```

Then include the tracker headers and feed detections from your detector each frame:

```cpp
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"

bytetrack::Config config;
bytetrack::ByteTrackTracker tracker(config);

std::vector<bytetrack::Detection> detections;
// Fill detection.xyxy, detection.conf, detection.cls, and detection.det_ind.
std::vector<bytetrack::TrackOutput> tracks = tracker.Update(detections, frame);
```

See [Native C++ Integration](../../../docs/native/index.md) for the public
native-backend workflow and embedding guidance.
