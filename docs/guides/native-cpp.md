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

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

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

If you need a C ABI instead of the C++ classes, build the shared target:

```bash
cmake -S boxmot/native/trackers/bytetrack -B build/native/bytetrack -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/bytetrack --config Release --target bytetrack_capi
```

The public ABI is declared in `boxmot/native/trackers/bytetrack/include/bytetrack/c_api.hpp`.
