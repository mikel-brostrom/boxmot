# Native C++ Integration

BoxMOT ships native C++ implementations of several trackers. You can use them in three ways:

1. From the CLI with `--tracker-backend cpp`.
2. From the Python facade with a method argument such as
   `model.track(..., tracker_backend="cpp")` or
   `model.val(..., tracker_backend="cpp")`; `model.tune(...)` accepts the same
   argument for cached replay.
3. Linked directly into your own C++ program via the `<tracker>_core` CMake
   target or the flat C ABI.

## Using the native backend from BoxMOT

Pass `--tracker-backend cpp` to swap the tracker implementation. It selects a
native live library for `track` and native cached replay for `eval` and `tune`:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval  --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --tracker-backend cpp
boxmot eval  --experiment mot17-ablation-yolox-lmbn --tracker botsort --tracker-backend cpp
```

`--tracking-backend cpp` is a compatibility alias for cached `eval` and `tune`;
live `track` uses `--tracker-backend cpp`. The `research` workflow currently
evaluates Python tracker code and does not forward either native selector.

In a source or editable install, the first live run builds the matching
`<tracker>_capi` shared library, while the first cached run builds the matching
`<tracker>_replay` executable. Both use `build/native/<tracker>/`. Use
`boxmot build` to prebuild the live C ABI libraries:

```bash
boxmot build                                          # native ReID + all live tracker libraries
boxmot build --tracker bytetrack --tracker ocsort     # subset
boxmot build --force                                  # rebuild even when artifacts already exist
```

`boxmot build` does not prebuild the cached replay executables; those are built
on first `eval` or `tune` use.

| Tracker | Live `track` | Cached replay | Notes |
| --- | --- | --- | --- |
| `botsort`    | Yes | Yes | AABB/OBB; uses native C++ ReID. |
| `bytetrack`  | Yes | Yes | AABB/OBB; no ReID. |
| `occluboost` | Yes | Yes | AABB/OBB; uses native C++ ReID for embeddings, recovery, and second pass. |
| `ocsort`     | Yes | Yes | AABB/OBB; no ReID. |
| `sfsort`     | Yes | Yes | AABB/OBB; no ReID. |

Every native tracker honors the `asso_func` value from its tracker
configuration. AABB and OBB tracking both support `iou`, `giou`, `diou`,
`ciou`, `hmiou`, and `centroid`. In OBB mode, overlap terms use oriented
intersections while enclosure and support terms use the corresponding
enclosing bounds required by each metric.

`centroid` normalizes distances by the frame dimensions. Live trackers infer
and cache those dimensions from the first image. SFSORT can instead use its
configured `frame_width` and `frame_height`; the other native trackers require
the initial image. Cached replay reads the image dimensions from the sequence.

Native live trackers do not currently support `per_class=True`. Use the Python
backend when each class needs separate tracker state.

### Native C++ ReID

When the selected tracker uses appearance features (currently `botsort` and
`occluboost`), `--tracker-backend cpp` also routes ReID embedding generation
through the native C++ ReID (`OnnxReIdModel`, exposed to Python as
`boxmot.native.reid.CppOnnxReID`) instead of the Python `ReID` backend. This
applies to live `track` and the cached `eval` / `tune` generate phase.

- If the supplied ReID weights are a `.pt` file, BoxMOT auto-exports a compatible ONNX artifact and reuses that export for later native runs.
- Embeddings are partitioned by their effective producer, model artifact, runtime, preprocessing, and crop semantics so incompatible results do not collide on disk.
- The native ReID runtime can be tuned through environment variables honoured
  by the wrappers and C++ runtime:
    - `BOXMOT_REID_BACKEND` — `auto` (default), `ort` / `onnxruntime`, or
      `opencv` / `dnn`. Auto prefers ONNX Runtime when it was compiled in and
      otherwise uses OpenCV DNN.
    - `BOXMOT_REID_DEVICE` — `auto` (default in the C++ runtime), `cpu`, `cuda`,
      or `coreml`. An unavailable accelerator provider falls back to CPU.

If the native ReID module is unavailable at backend-resolution time, BoxMOT
logs a warning and selects the Python producer before choosing a cache key.
Once the C++ producer has been selected, C ABI loading, model-loading, and
initialization failures are surfaced instead of silently switching producer.

### Embedding cache layout

Embedding caches use a producer-first layout:

```text
embs/
  <python|cpp>/
    <model>-<format>-<runtime>[-wHASH]/
      <preprocess>-cropvN/
        <sequence>.npy
```

For example, Python/PyTorch and C++/ONNX Runtime embeddings for the same source
checkpoint occupy different top-level producer and runtime buckets. The optional
`wHASH` token fingerprints the resolved model artifact, while `cropvN` versions
the crop geometry used before ReID preprocessing. Changing the producer, model
format or bytes, runtime, preprocessing mode, or crop schema therefore creates a
new bucket without invalidating compatible detection caches.

`python` and `cpp` identify the code path that actually produced the embeddings;
they do not identify the tracker algorithm. A C++ tracker requests the C++
producer. An import-time native-unavailable fallback is resolved to `python`
before cache lookup, while failures after C++ producer selection stop
generation. Multiple tracker algorithms may reuse one embedding bucket when
all producer and model semantics match.

Older caches may use a flat model bucket such as
`embs/<model>/<preprocess>/<sequence>.npy`. BoxMOT may reuse such a legacy file
only when it is explicitly considered compatible and trusted, is readable, and
has one embedding row per cached detection row. New or regenerated embeddings
are always written to the canonical producer-first layout. Do not reuse an
unidentified legacy bucket across model, runtime, producer, preprocessing, or
crop-schema changes.

The native replay path accepts both AABB benchmark caches and OBB caches. OBB replay outputs are written in the MMOT corner format expected by the OBB evaluation flow.

## Embedding native trackers in your own C++ program

Embed a BoxMOT native tracker in your own C++ program by linking against the tracker's `<tracker>_core` CMake target.

### Supported trackers

| Tracker | Directory | CMake target | Main class |
| --- | --- | --- | --- |
| ByteTrack  | `boxmot/native/cpp/trackers/bytetrack`  | `bytetrack_core`  | `bytetrack::ByteTrackTracker` |
| BoTSORT    | `boxmot/native/cpp/trackers/botsort`    | `botsort_core`    | `botsort::BotSortTracker` |
| OccluBoost | `boxmot/native/cpp/trackers/occluboost` | `occluboost_core` | `occluboost::OccluBoostTracker` |
| OCSORT     | `boxmot/native/cpp/trackers/ocsort`     | `ocsort_core`     | `ocsort::OCSortTracker` |
| SFSORT     | `boxmot/native/cpp/trackers/sfsort`     | `sfsort_core`     | `sfsort::SFSORTTracker` |

ReID for BoTSORT and OccluBoost is provided by the common static
`boxmot_tracker_base` target (`boxmot::trackers::base::OnnxReIdModel`) and is
pulled in transitively when you link against `<tracker>_core`.

> Calling from C, Rust, Go, Swift, JNI, .NET, etc.? Each tracker also exposes a flat C ABI in `boxmot/native/cpp/trackers/<tracker>/include/<tracker>/c_api.hpp` and produces a `<tracker>_capi.{so,dylib,dll}`. The header is the contract.

## Requirements

| Requirement | Minimum | Notes |
| --- | --- | --- |
| CMake | 3.16 | |
| C++17 compiler | GCC ≥ 7 / Clang ≥ 5 / AppleClang / MSVC ≥ 19.14 | |
| OpenCV | 4.x | Components: `calib3d core dnn imgcodecs imgproc video` |
| Eigen3 | 3.3 | Header-only |
| ONNX Runtime | 1.17+ | **Optional**, only for ReID (BoTSORT, OccluBoost) |

### Install system dependencies

=== "Ubuntu / Debian"

    ```bash
    sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev
    ```

=== "Fedora / RHEL"

    ```bash
    sudo dnf install -y gcc-c++ cmake opencv-devel eigen3-devel
    ```

=== "macOS"

    ```bash
    brew install cmake opencv eigen
    # Optional (ReID): brew install onnxruntime
    ```

=== "Windows (vcpkg)"

    ```powershell
    vcpkg install opencv4:x64-windows eigen3:x64-windows
    # Configure CMake with: -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
    ```

## Building from Python (`boxmot build`)

In a source or editable install, the CLI compiles native tracker libraries into
`build/native/<tracker>/`; no separate CMake invocation is needed. In a wheel,
prebuilt artifacts are installed beside their C++ sources under
`boxmot/native/cpp/trackers/<name>/`. See
[Using the native backend from BoxMOT](#using-the-native-backend-from-boxmot)
for the `boxmot build` commands and the distinction between live libraries and
cached replay executables.

## Minimal C++ project

Layout:

```text
native-demo/
├── CMakeLists.txt
└── main.cpp
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(boxmot_native_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOXMOT_ROOT "" CACHE PATH "Path to a BoxMOT source checkout")
if(NOT BOXMOT_ROOT)
    message(FATAL_ERROR "Pass -DBOXMOT_ROOT=/path/to/boxmot")
endif()

add_subdirectory(
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/bytetrack"
    "${CMAKE_BINARY_DIR}/boxmot_bytetrack")

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE bytetrack_core)
```

`main.cpp`:

```cpp
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"

#include <opencv2/core.hpp>
#include <iostream>

int main() {
    bytetrack::Config cfg;
    cfg.frame_rate   = 30;
    cfg.track_thresh = 0.5F;
    cfg.match_thresh = 0.8F;
    cfg.track_buffer = 30;
    cfg.asso_func    = "iou";

    bytetrack::ByteTrackTracker tracker(cfg);

    bytetrack::Detection det;
    det.xyxy << 100.0, 50.0, 200.0, 300.0;
    det.conf = 0.9F;
    det.cls = 0;
    det.det_ind = 0;

    for (const auto& t : tracker.Update({det}, cv::Mat{})) {
        std::cout << "id=" << t.id << " xyxy=("
                  << t.xyxy[0] << ", " << t.xyxy[1] << ", "
                  << t.xyxy[2] << ", " << t.xyxy[3] << ")\n";
    }
}
```

ByteTrack does not consume image pixels with the default IoU association, so
the example passes an empty `cv::Mat`. With `cfg.asso_func = "centroid"`, pass
the first frame so the tracker can cache its dimensions; later calls can again
use an empty matrix. The Python wrapper follows the same contract with
`tracker.update(dets, img)` for the initial centroid call and
`tracker.update(dets)` afterward.

Build and run:

```bash
cmake -S native-demo -B build/native-demo \
  -DBOXMOT_ROOT=/path/to/boxmot \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-demo
./build/native-demo/demo
```

If CMake can't find OpenCV/Eigen automatically, point at them explicitly:

```bash
-DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
-DEigen3_DIR=/path/to/eigen/share/eigen3/cmake
```

To use a different tracker, swap `bytetrack` for `botsort`, `ocsort`, `occluboost`, or `sfsort` (target name and namespace change accordingly).

## Detection contract

AABB:

```cpp
bytetrack::Detection det;
det.xyxy << x1, y1, x2, y2;
det.conf = confidence;
det.cls = class_id;
det.det_ind = detector_row_index;
```

OBB:

```cpp
bytetrack::Detection det;
det.is_obb = true;
det.xywha << cx, cy, w, h, angle_radians;
det.conf = confidence;
det.cls = class_id;
det.det_ind = detector_row_index;
```

Don't mix AABB and OBB on the same tracker instance — create a new one or call `Reset()` before switching.

## BoTSORT / OccluBoost ReID

Run without ReID via `cfg.with_reid = false`, or enable it by either:

- filling the `embedding` field on each detection from your own model, or
- setting `cfg.reid_model_path` to an ONNX model so the tracker computes embeddings via the bundled `OnnxReIdModel`.

Backend selection uses the same environment overrides as the Python wrappers:

- `BOXMOT_REID_BACKEND` — `auto`, `ort` / `onnxruntime`, or `opencv` / `dnn`
- `BOXMOT_REID_DEVICE` — `auto`, `cpu`, `cuda`, or `coreml`

ByteTrack, OCSORT, and SFSORT don't use ReID and are simpler to embed.
