# Native C++ Integration

Use this guide when you want to embed a BoxMOT native tracker in your own C++ (or C / Rust / Go / …) program instead of going through the Python CLI.

## Targets

Each tracker under `boxmot/native/trackers/<name>/` exposes two CMake targets:

| Target | Use it when … |
| --- | --- |
| `<tracker>_core` | You're writing C++ and want native types (`cv::Mat`, Eigen) |
| `<tracker>_capi` | You're calling from C, Rust, Go, Swift, JNI, .NET P/Invoke, or want to hot-swap the lib at runtime |

Supported trackers:

| Tracker | Directory | Core target | Main C++ class |
| --- | --- | --- | --- |
| ByteTrack  | `boxmot/native/trackers/bytetrack`  | `bytetrack_core`  | `bytetrack::ByteTrackTracker` |
| BoTSORT    | `boxmot/native/trackers/botsort`    | `botsort_core`    | `botsort::BotSortTracker` |
| OccluBoost | `boxmot/native/trackers/occluboost` | `occluboost_core` | `occluboost::OccluBoostTracker` |
| OCSORT     | `boxmot/native/trackers/ocsort`     | `ocsort_core`     | `ocsort::OCSortTracker` |
| SFSORT     | `boxmot/native/trackers/sfsort`     | `sfsort_core`     | `sfsort::SFSORTTracker` |

ReID for BoTSORT and OccluBoost is provided by the shared `boxmot_trackers_base` library (`boxmot::trackers::base::OnnxReIdModel`). Linking against `<tracker>_core` already pulls it in.

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

## Building

From a BoxMOT source checkout, either let the CLI do it:

```bash
boxmot build                                    # all trackers
boxmot build --tracker bytetrack --tracker ocsort   # subset
```

…or invoke CMake directly:

```bash
cmake -S boxmot/native/trackers/bytetrack -B build/native/bytetrack -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/bytetrack --config Release
```

Both produce the shared library next to the tracker sources:

| Platform | Output |
| --- | --- |
| Linux   | `bytetrack_capi.so` |
| macOS   | `bytetrack_capi.dylib` |
| Windows | `bytetrack_capi.dll` (+ `bytetrack_capi.lib`) |

## Option 1 — Link against the C++ core

Project layout:

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
    "${BOXMOT_ROOT}/boxmot/native/trackers/bytetrack"
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
#include <vector>

int main() {
    bytetrack::Config cfg;
    cfg.frame_rate   = 30;
    cfg.track_thresh = 0.5F;
    cfg.match_thresh = 0.8F;
    cfg.track_buffer = 30;

    bytetrack::ByteTrackTracker tracker(cfg);
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar::all(0));

    bytetrack::Detection det;
    det.xyxy << 100.0, 50.0, 200.0, 300.0;
    det.conf = 0.9F;
    det.cls = 0;
    det.det_ind = 0;

    const auto tracks = tracker.Update({det}, frame);
    for (const auto& t : tracks) {
        std::cout << "id=" << t.id << " xyxy=("
                  << t.xyxy[0] << ", " << t.xyxy[1] << ", "
                  << t.xyxy[2] << ", " << t.xyxy[3] << ")\n";
    }
}
```

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

### Detection contract

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

### BoTSORT / OccluBoost ReID

Run without ReID via `cfg.with_reid = false`, or enable it by either:

- filling the `embedding` field on each detection from your own model, or
- setting `cfg.reid_model_path` to an ONNX model so the tracker computes embeddings via the bundled `OnnxReIdModel`.

Backend selection mirrors the Python path through env vars:

- `BOXMOT_REID_BACKEND` — `ort` / `onnxruntime` (default) or `opencv` / `dnn`
- `BOXMOT_REID_DEVICE` — `cpu`, `cuda`, `coreml`, or `auto`

ByteTrack, OCSORT, and SFSORT don't use ReID and are simpler to embed.

## Option 2 — Link against the C ABI

The C ABI lives in `boxmot/native/trackers/<tracker>/include/<tracker>/c_api.hpp` and exposes the same five-function lifecycle for every tracker — substitute `bytetrack` for `botsort`, `ocsort`, `occluboost`, or `sfsort`:

```cpp
struct BoxMOTByteTrackConfig { /* tracker-specific knobs */ };
struct BoxMOTByteTrackHandle;

BoxMOTByteTrackHandle* boxmot_bytetrack_create(const BoxMOTByteTrackConfig*);
void                   boxmot_bytetrack_destroy(BoxMOTByteTrackHandle*);
int                    boxmot_bytetrack_reset(BoxMOTByteTrackHandle*);
int                    boxmot_bytetrack_update(
    BoxMOTByteTrackHandle* handle,
    const float* dets, int det_rows, int det_cols,    // (x1,y1,x2,y2,conf,cls) per row
    const std::uint8_t* image_data,
    int image_rows, int image_cols, int image_channels,
    float* out_tracks, int out_capacity, int* out_count);
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(boxmot_capi_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOXMOT_ROOT "" CACHE PATH "Path to a BoxMOT source checkout")
set(BYTETRACK_DIR "${BOXMOT_ROOT}/boxmot/native/trackers/bytetrack")

add_executable(capi_demo main.cpp)
target_include_directories(capi_demo PRIVATE "${BYTETRACK_DIR}/include")

if(WIN32)
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.lib")
elseif(APPLE)
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.dylib")
else()
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.so")
endif()

# So the loader finds the .so/.dylib at runtime without LD_LIBRARY_PATH.
set_target_properties(capi_demo PROPERTIES
    BUILD_RPATH   "${BYTETRACK_DIR}"
    INSTALL_RPATH "${BYTETRACK_DIR}")
```

`main.cpp`:

```cpp
#include "bytetrack/c_api.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    BoxMOTByteTrackConfig cfg{};
    cfg.min_conf     = 0.1f;
    cfg.track_thresh = 0.5f;
    cfg.match_thresh = 0.8f;
    cfg.track_buffer = 30;
    cfg.frame_rate   = 30;
    cfg.max_obs      = 50;

    auto* tracker = boxmot_bytetrack_create(&cfg);

    std::vector<float> dets = {100.f, 50.f, 200.f, 300.f, 0.9f, 0.f};   // x1,y1,x2,y2,conf,cls
    std::vector<std::uint8_t> img(720 * 1280 * 3, 0);

    // Output schema is 8 floats per track: (x1, y1, x2, y2, id, conf, cls, det_ind).
    std::vector<float> out(64 * 8);
    int n = 0;

    boxmot_bytetrack_update(
        tracker,
        dets.data(), 1, 6,
        img.data(), 720, 1280, 3,
        out.data(), static_cast<int>(out.size()), &n);

    for (int i = 0; i < n; ++i) {
        const float* r = &out[i * 8];
        std::printf("id=%.0f xyxy=(%.1f, %.1f, %.1f, %.1f)\n",
                    r[4], r[0], r[1], r[2], r[3]);
    }

    boxmot_bytetrack_destroy(tracker);
}
```

Build and run:

```bash
cmake -S capi-demo -B build/capi-demo \
    -DBOXMOT_ROOT=/path/to/boxmot \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build/capi-demo
./build/capi-demo/capi_demo
```
