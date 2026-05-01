# Native C++ Integration

Embed a BoxMOT native tracker in your own C++ program by linking against the tracker's `<tracker>_core` CMake target.

## Supported trackers

| Tracker | Directory | CMake target | Main class |
| --- | --- | --- | --- |
| ByteTrack  | `boxmot/native/trackers/bytetrack`  | `bytetrack_core`  | `bytetrack::ByteTrackTracker` |
| BoTSORT    | `boxmot/native/trackers/botsort`    | `botsort_core`    | `botsort::BotSortTracker` |
| OccluBoost | `boxmot/native/trackers/occluboost` | `occluboost_core` | `occluboost::OccluBoostTracker` |
| OCSORT     | `boxmot/native/trackers/ocsort`     | `ocsort_core`     | `ocsort::OCSortTracker` |
| SFSORT     | `boxmot/native/trackers/sfsort`     | `sfsort_core`     | `sfsort::SFSORTTracker` |

ReID for BoTSORT and OccluBoost is provided by the shared `boxmot_trackers_base` library (`boxmot::trackers::base::OnnxReIdModel`) and is pulled in transitively when you link against `<tracker>_core`.

> Calling from C, Rust, Go, Swift, JNI, .NET, etc.? Each tracker also exposes a flat C ABI in `boxmot/native/trackers/<tracker>/include/<tracker>/c_api.hpp` and produces a `<tracker>_capi.{so,dylib,dll}`. The header is the contract.

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

## Minimal project

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

    for (const auto& t : tracker.Update({det}, frame)) {
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

Backend selection mirrors the Python path through env vars:

- `BOXMOT_REID_BACKEND` — `ort` / `onnxruntime` (default) or `opencv` / `dnn`
- `BOXMOT_REID_DEVICE` — `cpu`, `cuda`, `coreml`, or `auto`

ByteTrack, OCSORT, and SFSORT don't use ReID and are simpler to embed.
