# Add a Tracker

To integrate a new tracker cleanly:

1. Add a module under the appropriate modality folder, such as `boxmot/trackers/bbox/<name>.py`, `boxmot/trackers/mask/<name>/`, or `boxmot/trackers/hybrid/<name>/`.
2. Implement a tracker class that subclasses `BaseTracker` and defines `update()`.
3. Add a `TrackerDefinition` to `TRACKER_DEFINITIONS` in
   `boxmot/trackers/registry.py`. `TRACKER_MAPPING`, `REID_TRACKERS`, and the
   class-name map are derived from those definitions.
4. Export the class from its modality package, such as
   `boxmot/trackers/bbox/__init__.py` or `boxmot/trackers/hybrid/__init__.py`.
   Add a higher-level re-export only when it is intentionally part of that
   package's public API.
5. Add `boxmot/configs/trackers/<name>.yaml` with each parameter's runtime default and tuning metadata.
6. Add a tracker doc page and wire it into `mkdocs.yml`.
7. Extend registry/package tests under `tests/unit/trackers/`, tracker-contract
   tests under `tests/unit/trackers/bbox/` or the relevant modality, and the
   tracker lists in `tests/test_config.py` where applicable.
8. Update the tracker, ReID, mask/OBB, and benchmark lists in
   `.github/workflows/` when the new tracker should run in those jobs.

## Optional native C++ backend

If the tracker also gets a native backend:

1. Add native sources under `boxmot/native/cpp/trackers/<name>/`.
2. Add the tracker subdirectory and wheel-install entries to
   `boxmot/native/cpp/CMakeLists.txt`.
3. Add Python wrapper code under `boxmot/native/trackers/<name>.py`.
4. Register live and replay backends in `boxmot/native/registry.py`.
5. Document `--tracker-backend cpp` support on the tracker page.
6. Add native wrapper tests under
   `tests/unit/native/trackers/test_native_<name>.py`.

Native tracker sources should follow the existing CMake layout: a
`<name>_replay` executable for cached `eval` and `tune`, a `<name>_capi` shared
library for live `track`, and a `<name>_core` target for reusable C++ code.

## Minimum checklist

- tracker implementation
- tracker registration
- tracker YAML
- docs page
- tests
- workflow matrices if benchmarked in CI
- native C++ registration and tests if a native backend is added
