# Native Tracker Layout

BoxMOT keeps Python and native tracker implementations in parallel trees:

- Python tracker runtime: `boxmot/trackers/<name>/`
- Native C++ tracker family: `boxmot/native/cpp/trackers/<name>/`
- Native C++ shared tracker base: `boxmot/native/cpp/trackers/base/`
- Python-side native registration: `boxmot/native/registry.py`

This keeps the user-facing tracker id stable while letting the backend vary by mode, while giving the C++ tree the same high-level shape as the Python tree: a shared base layer plus per-tracker packages.

## Organization

- `boxmot/native/cpp/trackers/base/`: shared C++ tracker abstractions and utilities analogous to the Python-side base tracker surface. Common helpers that are tracker-agnostic belong here.
- `boxmot/native/cpp/trackers/<name>/include/<name>/`: public tracker-specific headers.
- `boxmot/native/cpp/trackers/<name>/src/`: tracker-specific implementation and ABI entrypoints.

Tracker-specific code should stay in its own directory even when two trackers look similar. Only move code into `base/` when it is truly generic across tracker families.

## Selection

Prefer the dedicated tracker implementation selector:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker botsort --tracker-backend cpp
```

`--tracking-backend cpp` is still accepted as a compatibility alias for existing replay commands, but the canonical distinction is now:

- `--tracker-backend`: Python vs native C++ tracker implementation
- `--tracking-backend`: process/thread replay executor strategy

## Build entrypoints

- Per-tracker builds still work from `boxmot/native/cpp/trackers/<name>/` and are what the Python wrappers use today.
- The tree can also be configured from `boxmot/native/cpp/` to build the shared base and all registered native trackers together.

## Current scope

Native replay and live backends are currently registered for:

- `botsort`
- `bytetrack`
- `ocsort`
- `sfsort`

Each registered tracker exposes a native replay executable and an in-process live shared library, while Python continues to own backend selection and mode orchestration.

Build requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

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

See `docs/guides/native-cpp.md` for a complete CMake project and fake detector example.
