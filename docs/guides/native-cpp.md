# Native C++ Integration

Use this guide when you want to embed a BoxMOT native tracker in your own C++ program instead of calling the Python CLI or API.

## What You Compile

Each native tracker directory builds two useful targets:

- `<tracker>_core`: C++ classes such as `bytetrack::ByteTrackTracker`
- `<tracker>_capi`: shared C ABI used by the Python live wrapper

For a C++ application, prefer linking against `<tracker>_core` directly.

Supported native tracker directories:

| Tracker | Directory | Core target | Main C++ class |
| --- | --- | --- | --- |
| ByteTrack | `boxmot/native/trackers/bytetrack` | `bytetrack_core` | `bytetrack::ByteTrackTracker` |
| BoTSORT | `boxmot/native/trackers/botsort` | `botsort_core` | `botsort::BotSortTracker` |
| OccluBoost | `boxmot/native/trackers/occluboost` | `occluboost_core` | `occluboost::OccluBoostTracker` |
| OCSORT | `boxmot/native/trackers/ocsort` | `ocsort_core` | `ocsort::OCSortTracker` |
| SFSORT | `boxmot/native/trackers/sfsort` | `sfsort_core` | `sfsort::SFSORTTracker` |

ReID for ReID-using trackers (currently BoTSORT and OccluBoost) is provided by the shared library under `boxmot/native/trackers/base/` (target `boxmot_trackers_base`), which exposes `boxmot::trackers::base::OnnxReIdModel`. Linking against `<tracker>_core` already pulls in this dependency.

## Requirements

The same requirements apply whether you build via `boxmot build` (Python JIT
build) or directly with CMake from a C++ project.

| Requirement | Minimum version | Notes |
| --- | --- | --- |
| CMake | 3.16 | Used to configure and drive the build |
| C++17 compiler | GCC ≥ 7 / Clang ≥ 5 / AppleClang / MSVC ≥ 19.14 (VS 2017 15.7) | |
| OpenCV | 4.x | Components: `calib3d core dnn imgcodecs imgproc video` |
| Eigen3 | 3.3 | Header-only |
| ONNX Runtime | 1.17+ | **Optional.** Only needed for ReID-using trackers (BoTSORT, OccluBoost) |

### Install the system dependencies

=== "Ubuntu / Debian"

    ```bash
    sudo apt update
    sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev
    # Optional (ReID): install ONNX Runtime from the official release tarball
    # https://github.com/microsoft/onnxruntime/releases
    ```

=== "Fedora / RHEL"

    ```bash
    sudo dnf install -y gcc-c++ cmake opencv-devel eigen3-devel
    ```

=== "macOS (Homebrew)"

    ```bash
    brew install cmake opencv eigen
    # Optional (ReID):
    brew install onnxruntime
    ```

=== "Windows (vcpkg)"

    ```powershell
    vcpkg install opencv4:x64-windows eigen3:x64-windows
    # Optional (ReID):
    vcpkg install onnxruntime:x64-windows
    # Then configure CMake with: -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
    ```

### Verify the toolchain

```bash
cmake --version       # >= 3.16
c++ --version         # any C++17-capable compiler
pkg-config --modversion opencv4   # 4.x
```

If everything is in place you can build all native trackers in one go via:

```bash
boxmot build           # builds every registered native tracker
boxmot build --tracker bytetrack --tracker ocsort   # subset
```

## Compile a Tracker by Itself

From the BoxMOT repo root:

```bash
cmake -S boxmot/native/trackers/bytetrack -B build/native/bytetrack -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/bytetrack --config Release --target bytetrack_core
```

Swap `bytetrack` and `bytetrack_core` for another registered native tracker, for example `ocsort` / `ocsort_core`.

## Minimal Project

Create a small project outside the BoxMOT tree:

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
set(CMAKE_CXX_EXTENSIONS OFF)

set(BOXMOT_ROOT "" CACHE PATH "Path to a BoxMOT source checkout")
if(NOT BOXMOT_ROOT)
    message(FATAL_ERROR "Pass -DBOXMOT_ROOT=/path/to/boxmot")
endif()

add_subdirectory(
    "${BOXMOT_ROOT}/boxmot/native/trackers/bytetrack"
    "${CMAKE_BINARY_DIR}/boxmot_bytetrack"
)

add_executable(random_detector main.cpp)
target_link_libraries(random_detector PRIVATE bytetrack_core)
```

`main.cpp`:

```cpp
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <iostream>
#include <random>
#include <vector>

std::vector<bytetrack::Detection> RandomDetector(const cv::Mat& image, int frame_id) {
    std::mt19937 rng(1000 + frame_id);
    std::uniform_real_distribution<float> confidence(0.65F, 0.95F);

    std::vector<bytetrack::Detection> detections;
    detections.reserve(3);

    for (int index = 0; index < 3; ++index) {
        const double x1 = 80.0 + 140.0 * index + 3.0 * frame_id;
        const double y1 = 120.0 + 35.0 * index;
        const double width = 70.0;
        const double height = 120.0;

        if (x1 + width >= image.cols || y1 + height >= image.rows) {
            continue;
        }

        bytetrack::Detection detection;
        detection.xyxy << x1, y1, x1 + width, y1 + height;
        detection.conf = confidence(rng);
        detection.cls = 0;
        detection.det_ind = index;
        detections.push_back(detection);
    }

    return detections;
}

int main() {
    bytetrack::Config config;
    config.frame_rate = 30;
    config.track_thresh = 0.5F;
    config.match_thresh = 0.8F;
    config.track_buffer = 30;

    bytetrack::ByteTrackTracker tracker(config);

    for (int frame_id = 0; frame_id < 30; ++frame_id) {
        cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
        const std::vector<bytetrack::Detection> detections = RandomDetector(frame, frame_id);
        const std::vector<bytetrack::TrackOutput> tracks = tracker.Update(detections, frame);

        std::cout << "frame " << frame_id << ": " << tracks.size() << " tracks\n";
        for (const bytetrack::TrackOutput& track : tracks) {
            std::cout
                << "  id=" << track.id
                << " cls=" << track.cls
                << " conf=" << track.conf
                << " xyxy=("
                << track.xyxy[0] << ", "
                << track.xyxy[1] << ", "
                << track.xyxy[2] << ", "
                << track.xyxy[3] << ")\n";
        }
    }

    return 0;
}
```

Build and run it:

```bash
cmake -S native-demo -B build/native-demo \
  -DBOXMOT_ROOT=/path/to/boxmot \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/native-demo --config Release
./build/native-demo/random_detector
```

If CMake cannot find OpenCV or Eigen3 automatically, pass their package directories during configure:

```bash
cmake -S native-demo -B build/native-demo \
  -DBOXMOT_ROOT=/path/to/boxmot \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
  -DEigen3_DIR=/path/to/eigen/share/eigen3/cmake \
  -DCMAKE_BUILD_TYPE=Release
```

## Detection Contract

Your detector only needs to create tracker detections each frame.

For AABB detections:

```cpp
bytetrack::Detection detection;
detection.xyxy << x1, y1, x2, y2;
detection.conf = confidence;
detection.cls = class_id;
detection.det_ind = detector_row_index;
```

For OBB detections:

```cpp
bytetrack::Detection detection;
detection.is_obb = true;
detection.xywha << cx, cy, width, height, angle_radians;
detection.conf = confidence;
detection.cls = class_id;
detection.det_ind = detector_row_index;
```

Do not switch between AABB and OBB detections in the same tracker instance. Create a new tracker or call `Reset()` before changing detection geometry.

## BoTSORT / OccluBoost With ReID

BoTSORT and OccluBoost can run without ReID by setting `config.with_reid = false`, or they can use appearance features. Two equivalent paths are supported:

- Fill the `embedding` field on each detection yourself from your own ReID model.
- Or set `config.reid_model_path` to a native-compatible ONNX model so the tracker computes embeddings internally through the shared native `OnnxReIdModel`.

The internal ReID runtime can be selected at construction time, or via environment variables shared with the Python path:

- `BOXMOT_REID_BACKEND` — `ort` / `onnxruntime` (default) or `opencv` / `dnn` to use `cv2.dnn`.
- `BOXMOT_REID_DEVICE` — `cpu`, `cuda`, `coreml`, or `auto`.

For detector-only C++ examples, ByteTrack, OCSORT, and SFSORT are simpler because they do not require ReID inputs.

## Shared Library Option

If you need a C ABI instead of the C++ classes — for example to call into the
tracker from C, Rust, Go, or any language with `dlopen`/`LoadLibrary` FFI — link
against the `<tracker>_capi` shared library. Each tracker exposes the same
five-function lifecycle (`create`, `destroy`, `reset`, `update`, plus an output
buffer size helper) under an `extern "C"` block guarded by
`__declspec(dllexport)` (Windows) and `__attribute__((visibility("default")))`
(Linux/macOS).

### Build the library

```bash
cmake -S boxmot/native/trackers/bytetrack -B build/native/bytetrack -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/bytetrack --config Release --target bytetrack_capi
```

Or, if you have BoxMOT installed:

```bash
boxmot build --tracker bytetrack
```

This produces:

| Platform | Output |
| --- | --- |
| Linux   | `bytetrack_capi.so` |
| macOS   | `bytetrack_capi.dylib` |
| Windows | `bytetrack_capi.dll` |

next to the tracker's source directory under
`boxmot/native/trackers/bytetrack/`.

### Public C ABI header

The public ABI is declared in
`boxmot/native/trackers/bytetrack/include/bytetrack/c_api.hpp`. Every tracker
follows the same shape — substitute `bytetrack` for `botsort`, `ocsort`,
`occluboost`, or `sfsort`:

```cpp
struct BoxMOTByteTrackConfig { /* tracker-specific knobs */ };
struct BoxMOTByteTrackHandle;

BoxMOTByteTrackHandle* boxmot_bytetrack_create(const BoxMOTByteTrackConfig*);
void                   boxmot_bytetrack_destroy(BoxMOTByteTrackHandle*);
int                    boxmot_bytetrack_reset(BoxMOTByteTrackHandle*);
int                    boxmot_bytetrack_update(
    BoxMOTByteTrackHandle* handle,
    const float* dets, int det_rows, int det_cols,    /* (x1,y1,x2,y2,conf,cls) per row */
    const std::uint8_t* image_data,
    int image_rows, int image_cols, int image_channels,
    float* out_tracks, int out_capacity, int* out_count);
```

### Minimal CMake project

```text
capi-demo/
├── CMakeLists.txt
└── main.cpp
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(boxmot_capi_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOXMOT_ROOT "" CACHE PATH "Path to a BoxMOT source checkout")
if(NOT BOXMOT_ROOT)
    message(FATAL_ERROR "Pass -DBOXMOT_ROOT=/path/to/boxmot")
endif()

set(BYTETRACK_DIR "${BOXMOT_ROOT}/boxmot/native/trackers/bytetrack")

add_executable(capi_demo main.cpp)

target_include_directories(capi_demo PRIVATE
    "${BYTETRACK_DIR}/include")

# Link against the prebuilt shared lib produced by `boxmot build --tracker bytetrack`.
if(WIN32)
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.lib")
elseif(APPLE)
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.dylib")
else()
    target_link_libraries(capi_demo PRIVATE "${BYTETRACK_DIR}/bytetrack_capi.so")
endif()

# Embed the directory holding the tracker .so/.dylib in the executable's RPATH
# so the loader finds it at runtime without LD_LIBRARY_PATH gymnastics.
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
    cfg.min_conf      = 0.1f;
    cfg.track_thresh  = 0.5f;
    cfg.match_thresh  = 0.8f;
    cfg.track_buffer  = 30;
    cfg.frame_rate    = 30;
    cfg.max_obs       = 50;

    auto* tracker = boxmot_bytetrack_create(&cfg);

    // One detection: x1, y1, x2, y2, conf, cls.
    std::vector<float> dets = {100.f, 50.f, 200.f, 300.f, 0.9f, 0.f};
    std::vector<std::uint8_t> dummy_image(720 * 1280 * 3, 0);

    // Output schema is (x1, y1, x2, y2, id, conf, cls, det_ind) — 8 floats per track.
    std::vector<float> out_tracks(64 * 8);
    int out_count = 0;

    boxmot_bytetrack_update(
        tracker,
        dets.data(), 1, 6,
        dummy_image.data(), 720, 1280, 3,
        out_tracks.data(), static_cast<int>(out_tracks.size()), &out_count);

    std::printf("got %d track(s)\n", out_count);
    for (int i = 0; i < out_count; ++i) {
        const float* row = &out_tracks[i * 8];
        std::printf("  id=%.0f conf=%.2f xyxy=(%.1f, %.1f, %.1f, %.1f)\n",
                    row[4], row[5], row[0], row[1], row[2], row[3]);
    }

    boxmot_bytetrack_destroy(tracker);
    return 0;
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

### Choosing between the C++ core and the C ABI

| Need | Recommended target |
| --- | --- |
| Pure-C++ project, want `cv::Mat` / Eigen types | `<tracker>_core` (see [Minimal Project](#minimal-project)) |
| Calling from C, Rust, Go, Swift, JNI, .NET P/Invoke | `<tracker>_capi` (this section) |
| Hot-swapping the tracker library at runtime | `<tracker>_capi` |
| Smallest possible link-time dependency surface | `<tracker>_capi` |
