# Add a Tracker

To integrate a new tracker cleanly:

1. Add a module under `boxmot/trackers/<name>/`.
2. Implement a tracker class that subclasses `BaseTracker` and defines `update()`.
3. Register it in `boxmot/trackers/tracker_zoo.py`.
4. Export it from `boxmot/trackers/__init__.py` and `boxmot/__init__.py`.
5. Add a default YAML under `boxmot/configs/trackers`.
6. Add a tracker doc page and wire it into `mkdocs.yml`.
7. Extend tests and CI matrices where needed.

## Minimum checklist

- tracker implementation
- tracker registration
- tracker YAML
- docs page
- tests
- workflow matrices if benchmarked in CI
