# Add a Tracker

To integrate a new tracker cleanly:

1. Add a module under the appropriate modality folder, such as `boxmot/trackers/bbox/<name>.py`, `boxmot/trackers/mask/<name>/`, or `boxmot/trackers/hybrid/<name>/`.
2. Implement a tracker class that subclasses `BaseTracker` and defines `update()`.
3. Register it in `boxmot/trackers/registry.py`.
4. Export it from `boxmot/trackers/__init__.py` and `boxmot/__init__.py`.
5. Add a default YAML under `boxmot/configs/trackers`.
6. Add a tracker doc page and wire it into `mkdocs.yml`.
7. Extend tests and CI matrices where needed.

## Optional native C++ backend

If the tracker also gets a native backend:

1. Add native sources under `boxmot/native/cpp/trackers/<name>/`.
2. Add Python wrapper code under `boxmot/native/trackers/<name>.py`.
3. Register live and replay backends in `boxmot/native/registry.py`.
4. Document `--tracker-backend cpp` support on the tracker page.
5. Add native wrapper tests under `tests/unit/test_native_<name>.py`.

Native tracker sources should follow the existing CMake layout: a replay executable for cached benchmark modes and a shared library for live `track`.

## Minimum checklist

- tracker implementation
- tracker registration
- tracker YAML
- docs page
- tests
- workflow matrices if benchmarked in CI
- native C++ registration and tests if a native backend is added
