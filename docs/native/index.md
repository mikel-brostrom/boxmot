# Native tracker backends

BoxMOT provides C++ backends for BotSort, ByteTrack, OccluBoost, OcSort, and
SFSORT. Native trackers implement the same structured tracker contract as the
Python backends:

`cpp` is a runtime backend, not a tracker representation family. These
implementations remain box trackers in the domain taxonomy, while their C++
sources and typed ABI stay under `boxmot/native/cpp`.

```python
from boxmot import create_tracker
from boxmot.structures import Boxes, Detections, Frame
from boxmot.trackers import TrackerSpec

import torch

tracker = create_tracker(
    TrackerSpec(name="bytetrack", backend="cpp", geometry="aabb")
)
frame = Frame(
    image=torch.zeros((3, 720, 1280), dtype=torch.uint8),
    sample_id="camera-1:000001",
    sequence_id="camera-1",
    frame_index=1,
)
detections = Detections(
    geometry=Boxes(torch.tensor([[10, 20, 80, 160]], dtype=torch.float32)),
    scores=torch.tensor([0.9], dtype=torch.float32),
    class_ids=torch.tensor([0], dtype=torch.int64),
    sample_id=frame.sample_id,
)
tracks = tracker.update(detections, frame)
```

Canonical inputs and outputs remain CPU-contiguous Torch structures at the
Python boundary. For simple box-only calls, the high-level native tracker also
accepts the same exact NumPy AABB6 or OBB7 matrix as the Python backend and
returns packed `float64` AABB8 or OBB9 rows. Use `Detections` when returning
`Tracks` or providing enrichments and sample metadata. The optional `frame`
accepts either a canonical `Frame` with an RGB CHW tensor or a `uint8` NumPy
image with shape `(height, width, 3)` in BGR order. Strided NumPy images are
made contiguous when needed.

The low-level ctypes modules under `boxmot/native/trackers/` accept only typed,
contiguous NumPy buffers. Canonical conversion, requirements, configuration,
reset behavior, and OBB angle continuity live with each algorithm in
`boxmot/trackers/<name>/native.py`. Backend validation and construction use
the canonical factory in `boxmot/trackers/common/factory.py`.

## Capabilities and requirements

| Tracker | AABB | OBB | Embeddings | Frame |
| --- | --- | --- | --- | --- |
| `botsort` | Yes | Yes | When `use_embeddings` | CMC or centroid association |
| `bytetrack` | Yes | Yes | No | Centroid association |
| `occluboost` | Yes | Yes | When `use_embeddings` | CMC or centroid association |
| `ocsort` | Yes | Yes | No | Centroid association |
| `sfsort` | Yes | Yes | No | Dimensions, unless width and height are configured |

Requirements are frozen when the tracker is created and are available from
`tracker.requirements`. Supply a frame on every update when it is required,
including for centroid association. SFSORT can run without a frame when both
positive `frame_width` and `frame_height` are configured. Otherwise it needs
the frame dimensions, but does not consume its pixels. A pipeline supplies the requested frame. It may enrich
detections with a shared appearance encoder, or a ReID-enabled native adapter
may derive missing embeddings privately before invoking its C++ library.

Native trackers do not support masks or per-class tracker state. Their factory
rejects those modes, unknown tracker options, model/weight options placed in
`TrackerSpec`, and geometry modes unsupported by the selected native
implementation. Configure tracker-owned ReID separately with
`tracker.configure_reid(spec)`.

Native trackers retain fixed-step prediction and reject `variable_dt=True`.
See the [tracker input matrix](../trackers/index.md#input-support) for the Python
implementations and their additional input modes.

The high-level BotSort and OccluBoost native adapters consume embeddings already
present on `Detections.embeddings` or lazily derive missing embeddings from a
supplied `Frame` or NumPy image. In either case, their C++ tracker libraries
receive only typed feature buffers and never load, export, download, or run a
ReID model. Reusable native ReID inference remains a separate appearance-encoder
concern and is not linked into tracker libraries. Its component-facing adapter
lives with the ReID backends; `boxmot.native` contains only the C++ sources,
build/load support, and low-level typed bindings. Native ReID accepts a resolved
ONNX artifact and never performs an implicit download or conversion.

All trackers support `iou`, `giou`, `diou`, `ciou`, `hmiou`, and `centroid`
association for AABB and OBB geometry. OBB inputs use `(cx, cy, w, h, angle)`
with an unwrapped angle in radians. One tracker instance has a fixed geometry
mode; use `reset()` for a new sequence, not to change its mode.

## Building

In a source or editable install, BoxMOT resolves or builds the matching
`<tracker>_capi` shared library. Prebuild all registered native libraries with:

```bash
boxmot build
```

Build a subset with repeated tracker selectors:

```bash
boxmot build --tracker bytetrack --tracker ocsort
```

The native tree requires CMake 3.16+, a C++17 compiler, OpenCV 4.x, and Eigen
3.3+. Per-tracker CMake builds remain available:

```bash
cmake -S boxmot/native/cpp/trackers/bytetrack \
  -B build/native/bytetrack \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/bytetrack --target bytetrack_capi
```

Each tracker links the model-free `boxmot_tracker_base` target, which contains
only association and assignment code. A per-tracker configure does not discover
or directly link OpenCV DNN, ONNX Runtime, or the native ReID implementation.

The independent `reid_capi` library and `boxmot_native_reid` target live under
`boxmot/native/cpp/reid`. They are enabled by default for the aggregate native
build, and can be excluded from a tracker-only aggregate build explicitly:

```bash
cmake -S boxmot/native/cpp \
  -B build/native/all-trackers \
  -DBOXMOT_BUILD_NATIVE_REID=OFF
cmake --build build/native/all-trackers
```

Only `boxmot_native_reid` explicitly discovers and links OpenCV DNN and, when
available, ONNX Runtime. None of the tracker C API libraries link that target.
An OpenCV distribution may itself give a tracker-required module, such as
`video`, additional transitive system dependencies; BoxMOT does not request
those modules as tracker dependencies.

The native appearance encoder can also be built independently:

```bash
cmake -S boxmot/native/cpp/reid \
  -B build/native/reid \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/reid --target reid_capi
```

There are no native replay executables. Evaluation and tuning stream keyed
Parquet records through the same live tracker API used for ordinary tracking.
Legacy positional NPY/NPZ caches are unsupported.

## Typed C ABI v2

Each tracker exports a flat C ABI from
`boxmot/native/cpp/trackers/<name>/include/<name>/c_api.hpp`. Shared v2 buffer
types live in
`boxmot/native/cpp/trackers/base/include/boxmot/trackers/base/c_api_v2.hpp`.

The update function has this shape, with `<name>` replaced by the tracker ID:

```cpp
int boxmot_<name>_update_v2(
    BoxMOT<Name>Handle* handle,
    const BoxMOTDetectionBatchV2* detections,
    const BoxMOTImageV2* image,
    BoxMOTTrackBatchV2** output);

void boxmot_<name>_result_free_v2(BoxMOTTrackBatchV2* output);
```

`BoxMOTDetectionBatchV2` is columnar:

- `geometry`: contiguous `float[rows * geometry_cols]`; four AABB columns or
  five OBB columns.
- `scores`: contiguous `float[rows]`.
- `class_ids`: contiguous `int64_t[rows]`.
- `detection_indices`: contiguous `int64_t[rows]`.
- `embeddings`: optional contiguous `float[rows * embedding_cols]`.

The output keeps geometry and scores in float buffers and track IDs, class IDs,
and detection indices in separate `int64_t` buffers. This avoids precision loss
from routing identifiers through a float row matrix.

### Ownership and update semantics

On success, `update_v2` stores a library-allocated result object in `output`,
including for an empty result. The caller must invoke the matching
`result_free_v2` exactly once after copying or consuming the buffers.

The output size is determined after the tracker update. Callers do not provide
capacity and must never retry an update to resize a buffer: an update mutates
tracker state and is performed exactly once. On failure, the function returns
zero and the error text is available from `boxmot_<name>_last_error()`.

### Detection and track layouts

The ABI carries fields in typed buffers, not legacy row layouts. When an
explicit serializer is needed at a file or wire boundary, BoxMOT uses:

- AABB detections: `(x1, y1, x2, y2, score, class)`.
- OBB detections: `(cx, cy, w, h, angle, score, class)`.
- AABB tracks: `(x1, y1, x2, y2, id, score, class, detection_index)`.
- OBB tracks: `(cx, cy, w, h, angle, id, score, class, detection_index)`.

A detection index of `-1` identifies a propagated track without a matching
detection in the current frame.

## Linking the C++ core directly

Applications written in C++ may link the tracker core instead of the C ABI:

```cmake
add_subdirectory(
  "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/bytetrack"
  "${CMAKE_BINARY_DIR}/boxmot_bytetrack")
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE bytetrack_core)
```

The core API accepts typed `Detection` objects. Appearance-capable trackers
expect the caller to populate `Detection.embedding`; they do not own an
inference backend.
